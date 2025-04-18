from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, PaliGemmaProcessor, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import torch
import numpy as np
import math
# Import the ModelManager from inference.py
from inference import model_manager, set_models_directory, run_inference, Models
# Import utility functions from utils.py
from utils import load_image_tensor, convert_tensor_to_pil, convert_pil_to_tensor


# Default model cache directory
DEFAULT_CACHE_DIR = "/blue/rcstudents/luke.sutor/adversarial-vlms/models/"


def get_token_id(tokenizer, string):
    # Gets the token ID of the specified string
    tokens = tokenizer.encode(string, add_special_tokens=False)
    if len(tokens) == 0:
        raise ValueError(f"The string '{string}' could not be tokenized.")
    return tokens[0]


def get_generation(model_id, prompt, image_path):
    """
    Returns the generated text from passing an image through a model.
    Uses the model manager to load the appropriate model and processor.
    
    Args:
        model_id: HuggingFace model identifier
        prompt: Text prompt for the model
        image_path: Path to the image file
    
    Returns:
        Generated text from the model
    """
    # Use the run_inference function from inference.py
    return run_inference(model_id, prompt, image_path, max_new_tokens=100)


# ==============================================
# Attack Functions Registry
# ==============================================

class AttackRegistry:
    """Registry for model-specific PGD attack implementations."""
    
    _registry = {}
    
    @classmethod
    def register(cls, model_type):
        """Decorator to register a model-specific attack function"""
        def inner_wrapper(wrapped_class):
            cls._registry[model_type] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def get_attack_function(cls, model_id):
        """Get the appropriate attack function for a model"""
        # Check for exact model match first
        if model_id in cls._registry:
            return cls._registry[model_id]
        
        # Check for model family match
        if "paligemma" in model_id.lower():
            return cls._registry.get("paligemma", cls._registry.get("default"))
        elif "qwen2.5-vl" in model_id.lower():
            return cls._registry.get("qwen25", cls._registry.get("qwen", cls._registry.get("default")))
        elif "qwen2-vl" in model_id.lower():
            return cls._registry.get("qwen", cls._registry.get("default"))
        elif "llama-3.2" in model_id.lower() and "vision" in model_id.lower():
            return cls._registry.get("llama", cls._registry.get("default"))
        elif "llava" in model_id.lower():
            return cls._registry.get("llava", cls._registry.get("default"))
        
        # Fall back to default
        return cls._registry.get("default")


# ==============================================
# Alpha Scheduler Implementation
# ==============================================

class AlphaScheduler:
    """Implements different scheduling strategies for alpha parameter in PGD attacks."""
    
    @staticmethod
    def get_scheduler(scheduler_type="cosine"):
        """Factory method to get the appropriate scheduler function"""
        schedulers = {
            "linear": AlphaScheduler.linear_schedule,
            "cosine": AlphaScheduler.cosine_schedule,
            "polynomial": AlphaScheduler.polynomial_schedule,
        }
        return schedulers.get(scheduler_type, AlphaScheduler.cosine_schedule)
    
    @staticmethod
    def linear_schedule(iteration, num_iterations, warmup_ratio, alpha_max, alpha_min=0.0):
        """Linear warmup and linear decay schedule"""
        warmup_steps = int(num_iterations * warmup_ratio)
        
        if iteration < warmup_steps:
            # Linear warmup
            return alpha_min + (alpha_max - alpha_min) * (iteration / max(1, warmup_steps))
        else:
            # Linear decay
            decay_ratio = max(0.0, (num_iterations - iteration) / max(1, (num_iterations - warmup_steps)))
            return alpha_min + (alpha_max - alpha_min) * decay_ratio
    
    @staticmethod
    def cosine_schedule(iteration, num_iterations, warmup_ratio, alpha_max, alpha_min=0.0):
        """Linear warmup and cosine decay schedule"""
        warmup_steps = int(num_iterations * warmup_ratio)
        
        if iteration < warmup_steps:
            # Linear warmup
            return alpha_min + (alpha_max - alpha_min) * (iteration / max(1, warmup_steps))
        else:
            # Cosine decay
            decay_steps = num_iterations - warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (iteration - warmup_steps) / max(1, decay_steps)))
            return alpha_min + (alpha_max - alpha_min) * cosine_decay
    
    @staticmethod
    def polynomial_schedule(iteration, num_iterations, warmup_ratio, alpha_max, alpha_min=0.0, power=2.0):
        """Linear warmup and polynomial decay schedule"""
        warmup_steps = int(num_iterations * warmup_ratio)
        
        if iteration < warmup_steps:
            # Linear warmup
            return alpha_min + (alpha_max - alpha_min) * (iteration / max(1, warmup_steps))
        else:
            # Polynomial decay
            decay_steps = num_iterations - warmup_steps
            decay_ratio = (1 - (iteration - warmup_steps) / max(1, decay_steps)) ** power
            return alpha_min + (alpha_max - alpha_min) * decay_ratio


# ==============================================
# Base Attack Implementation
# ==============================================

class BasePGDAttack:
    """Base class for PGD attack implementations."""
    
    def __init__(self, model_id, epsilon=0.02, alpha_max=0.01, alpha_min=0.0001, 
                 num_iter=10, token_lookahead=3, warmup_ratio=0.1, 
                 scheduler_type="cosine", return_best=True):
        self.model_id = model_id
        self.epsilon = epsilon
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.num_iter = num_iter
        self.token_lookahead = token_lookahead
        self.warmup_ratio = warmup_ratio
        self.scheduler_type = scheduler_type
        self.return_best = return_best
        
        # Get the alpha scheduler function
        self.alpha_scheduler = AlphaScheduler.get_scheduler(scheduler_type)
        
        # Load model, tokenizer, and processor via model manager
        self.model, self.tokenizer, self.processor = model_manager.get_model(model_id)
    
    def get_alpha_for_iteration(self, iteration):
        """Calculate alpha value for the current iteration based on schedule"""
        return self.alpha_scheduler(
            iteration=iteration,
            num_iterations=self.num_iter,
            warmup_ratio=self.warmup_ratio,
            alpha_max=self.alpha_max,
            alpha_min=self.alpha_min
        )
    
    def process_inputs(self, prompt, image):
        """Process inputs based on model type - to be implemented by subclasses"""
        raise NotImplementedError
    
    def generate_from_inputs(self, inputs, max_new_tokens=10):
        """Generate text from processed inputs - to be implemented by subclasses"""
        raise NotImplementedError
    
    def update_working_inputs(self, working_inputs, target_id):
        """Update inputs for next token prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def pgd_attack(self, prompt, image_path, target_sequence):
        """
        Advanced PGD attack that optimizes an image to make a vision-language model generate
        a target sequence by iteratively optimizing for each token in the sequence.
        
        Args:
            prompt: Text prompt to use with the image
            image_path: Path to the input image
            target_sequence: Target text sequence to generate
        
        Returns:
            Tuple of (numpy_image, tensor_image) where:
            - numpy_image: numpy array of shape [H, W, C] in range [0, 1]
            - tensor_image: PyTorch tensor of shape [1, C, H, W]
            
        Note: The calling function pgd_attack() will only return tensor_image.
        """
        # Process the image and text prompt
        image = Image.open(image_path).convert("RGB")
        model_inputs = self.process_inputs(prompt, image)
        
        # Set up for gradient tracking
        processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
        model_inputs['pixel_values'] = processed_image
        original_image = processed_image.clone()
        
        # Convert target sequence to token IDs
        target_token_ids = self.tokenizer.encode(target_sequence, add_special_tokens=False)
        print(f"Target tokens: {[self.tokenizer.decode([tid]) for tid in target_token_ids]}")
        
        # Initialize momentum for more stable gradients
        momentum = torch.zeros_like(processed_image.data)

        # Initialize noise standard deviation
        noise_std = 0.01
        
        # Track best result
        best_image = processed_image.clone()
        best_prob = 0.0
        
        # Print scheduler information
        print(f"Using {self.scheduler_type} alpha scheduler with warmup ratio {self.warmup_ratio}")
        print(f"Alpha range: {self.alpha_min:.6f} to {self.alpha_max:.6f}")
        
        for i in range(self.num_iter):
            # Calculate current alpha based on schedule
            current_alpha = self.get_alpha_for_iteration(i)
            
            print(f"=== Iteration {i+1}/{self.num_iter} (alpha={current_alpha:.6f}) ===")

            # Add random noise to improve robustness against quantization
            with torch.no_grad():
                noise = torch.randn_like(processed_image) * noise_std
                noisy_image = processed_image.clone() + noise
                noisy_image = torch.clamp(noisy_image, 0, 1)
            
            # Create new inputs with noisy image
            noisy_inputs = {k: v.clone() for k, v in model_inputs.items()}
            noisy_inputs['pixel_values'] = noisy_image.requires_grad_(True)
            
            # Accumulate gradients over multiple tokens
            self.model.zero_grad()
            total_loss = 0
            
            # Create a working copy of inputs that we'll modify for sequential prediction
            working_inputs = {k: v.clone() for k, v in noisy_inputs.items()}
            
            # Number of tokens to optimize (limited by target sequence length)
            n_tokens = min(self.token_lookahead, len(target_token_ids))
            
            # For each token position in the sequence (up to lookahead)
            for j in range(n_tokens):
                # Get target token for this position
                target_id = target_token_ids[j]
                
                # Forward pass to get logits
                outputs = self.model(**working_inputs)
                
                # Get logits for the last position (predicting next token)
                position_logits = outputs.logits[0, -1]
                
                # Calculate loss for this token
                log_probs = torch.log_softmax(position_logits, dim=-1)
                token_loss = -log_probs[target_id]  # Negative log likelihood

                # Position weighting - focus strongly on first token with linear decay for subsequent tokens
                # First token gets weight=token_lookahead, second gets token_lookahead-1, etc.
                position_weight = max(float(self.token_lookahead - j), 0.5)  # Ensure minimum weight of 0.5
                
                # Add weighted loss to total
                weighted_loss = token_loss * position_weight
                total_loss = total_loss + weighted_loss
                
                # Print probabilities for monitoring
                probs = torch.softmax(position_logits, dim=-1)
                print(f"  Position {j}: Target '{self.tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
                top_token_id = torch.argmax(position_logits).item()
                print(f"    Top predicted: '{self.tokenizer.decode([top_token_id])}' prob: {probs[top_token_id].item():.4f}")
                
                # For next token prediction, append this token to input
                # Skip this step for the last token we're optimizing
                if j < n_tokens - 1:
                    # Update working inputs for next token prediction
                    self.update_working_inputs(working_inputs, target_id)
            
            # Backpropagate the total loss
            total_loss.backward()
            
            # Track first token probability for best result
            first_token_prob = probs[target_token_ids[0]].item()
            if first_token_prob > best_prob:
                best_prob = first_token_prob
                best_image = processed_image.clone()

            # PGD update with momentum
            if noisy_inputs['pixel_values'].grad is not None:
                # Update momentum term (helps stabilize optimization)
                momentum = 0.9 * momentum - current_alpha * torch.sign(noisy_inputs['pixel_values'].grad)
                
                # Apply momentum update
                processed_image = processed_image + momentum
                
                # Project back to epsilon ball
                perturbation = torch.clamp(processed_image - original_image, -self.epsilon, self.epsilon)
                processed_image = original_image + perturbation
                
                # Quantize directly to valid 8-bit values (multiples of 1/255)
                with torch.no_grad():
                    # Round to nearest 8-bit pixel values
                    # processed_image = torch.round(processed_image * 255) / 255
                    
                    # Ensure values stay in valid range after rounding
                    processed_image = torch.clamp(processed_image, 0, 1)
                    
                    # Since we're using discrete pixel values, small noise is still useful
                    # but we can use a smaller fixed value
                    noise_std = 0.003
                    
                    # Update model inputs with properly quantized image
                    model_inputs['pixel_values'] = processed_image.clone()
                    
                    # Prepare for next iteration
                    processed_image = processed_image.detach().requires_grad_(True)
            else:
                print("Warning: No gradient calculated.")
            
            print()

        # Use the best image found during optimization
        if best_prob > 0.0 and self.return_best:
            print(f"\nUsing best image with probability {best_prob:.4f}")
            processed_image = best_image
        else:
            print(f"\nUsing final image")

        # Save the tensor version before any transformations
        tensor_image = processed_image.clone()

        # print("shape:", tensor_image.shape) torch.Size([1, 3, 224, 224])
        
        return tensor_image


# ==============================================
# Model-Specific Attack Implementations
# ==============================================

@AttackRegistry.register("paligemma")
class PaliGemmaAttack(BasePGDAttack):
    """PGD attack implementation for PaliGemma models."""
    
    def process_inputs(self, prompt, image):
        """Process inputs for PaliGemma models"""
        return self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
    
    def generate_from_inputs(self, inputs, max_new_tokens=10):
        """Generate text from processed inputs for PaliGemma models"""
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = self.processor.decode(
                generation[0][input_len:], skip_special_tokens=True
            )
        return generated_text
    
    def update_working_inputs(self, working_inputs, target_id):
        """Update inputs for next token prediction for PaliGemma models"""
        # Add the target token to the input sequence
        new_token_tensor = torch.tensor([[target_id]], device=working_inputs['input_ids'].device)
        working_inputs['input_ids'] = torch.cat([working_inputs['input_ids'], new_token_tensor], dim=1)
        
        # Also update attention mask if present
        if 'attention_mask' in working_inputs:
            working_inputs['attention_mask'] = torch.cat(
                [working_inputs['attention_mask'], 
                 torch.ones((1, 1), device=working_inputs['attention_mask'].device)], 
                dim=1
            )


@AttackRegistry.register("qwen")
class QwenAttack(BasePGDAttack):
    """PGD attack implementation for Qwen models."""
    
    def process_inputs(self, prompt, image):
        """Process inputs for Qwen models"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        return inputs
    
    def generate_from_inputs(self, inputs, max_new_tokens=10):
        """Generate text from processed inputs for Qwen models"""
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text[0] if output_text else ""
    
    def update_working_inputs(self, working_inputs, target_id):
        """Update inputs for next token prediction for Qwen models"""
        # Add the target token to the input sequence
        new_token_tensor = torch.tensor([[target_id]], device=working_inputs['input_ids'].device)
        working_inputs['input_ids'] = torch.cat([working_inputs['input_ids'], new_token_tensor], dim=1)
        
        # Also update attention mask
        working_inputs['attention_mask'] = torch.cat(
            [working_inputs['attention_mask'], 
             torch.ones((1, 1), device=working_inputs['attention_mask'].device)], 
            dim=1
        )


# Register a default implementation that will work with basic models
@AttackRegistry.register("default")
class DefaultAttack(PaliGemmaAttack):
    """Default PGD attack implementation for most models."""
    pass

# Register the Qwen2.5 attack class (uses same implementation as Qwen since they have the same interface)
@AttackRegistry.register("qwen25")
class Qwen25Attack(QwenAttack):
    """PGD attack implementation for Qwen 2.5 VL models."""
    # Inherits all functionality from QwenAttack since the interfaces are compatible
    pass

@AttackRegistry.register("llama")
class LlamaAttack(BasePGDAttack):
    """PGD attack implementation for Llama 3.2 Vision models."""
    
    def process_inputs(self, prompt, image):
        """Process inputs for Llama Vision models"""
        # Format as chat message
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Process inputs
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        
        return inputs
    
    def update_working_inputs(self, working_inputs, target_id):
        """Update inputs for next token prediction for Llama models"""
        # Add the target token to the input sequence
        if 'input_ids' in working_inputs:
            new_token_tensor = torch.tensor([[target_id]], device=working_inputs['input_ids'].device)
            working_inputs['input_ids'] = torch.cat([working_inputs['input_ids'], new_token_tensor], dim=1)
            
            # Also update attention mask if present
            if 'attention_mask' in working_inputs:
                working_inputs['attention_mask'] = torch.cat(
                    [working_inputs['attention_mask'], 
                    torch.ones((1, 1), device=working_inputs['attention_mask'].device)], 
                    dim=1
                )

    def pgd_attack(self, prompt, image_path, target_sequence):
        """
        PGD attack for Llama 3.2 Vision models that handles the unique multi-crop tensor structure
        
        Args:
            prompt: Text prompt to use with the image
            image_path: Path to the input image
            target_sequence: Target text sequence to generate
        
        Returns:
            PyTorch tensor containing the optimized image with shape [1, 3, H, W]
        """
        # Process the image and text prompt
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # Save original size for reference
        print(f"Original image size: {original_size}")
        
        # Get model inputs which will have the complex tensor structure
        model_inputs = self.process_inputs(prompt, image)
        
        if 'pixel_values' not in model_inputs:
            raise ValueError(f"Llama inputs must contain 'pixel_values', found keys: {list(model_inputs.keys())}")
        
        # Analyze the tensor structure
        pixel_shape = model_inputs['pixel_values'].shape
        print(f"Model's pixel_values shape: {pixel_shape}")
        
        # For reference, expected shape is [1, 1, 4, 3, 560, 560]
        # batch, extra_dim, num_crops, channels, height, width
        
        # Create a smaller working tensor with just the first crop for gradient tracking
        # Extract the first crop (index 0)
        if len(pixel_shape) >= 4 and pixel_shape[2] > 1:
            # Structure is [1, 1, 4, 3, H, W] - extract first crop
            base_image = model_inputs['pixel_values'][:, :, 0].clone().detach().requires_grad_(True)
            print(f"Working with first crop, shape: {base_image.shape}")
        else:
            # Fallback if structure is different
            base_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
        
        # Save original for epsilon constraint
        original_base_image = base_image.clone()
        
        # Convert target sequence to token IDs
        target_token_ids = self.tokenizer.encode(target_sequence, add_special_tokens=False)
        print(f"Target tokens: {[self.tokenizer.decode([tid]) for tid in target_token_ids]}")
        
        # Initialize momentum for more stable gradients
        momentum = torch.zeros_like(base_image)
        
        # Initialize noise and tracking variables
        noise_std = 0.01
        best_image = None
        best_prob = 0.0
        
        # Print scheduler information
        print(f"Using {self.scheduler_type} alpha scheduler with warmup ratio {self.warmup_ratio}")
        print(f"Alpha range: {self.alpha_min:.6f} to {self.alpha_max:.6f}")
        
        for i in range(self.num_iter):
            # Calculate current alpha based on schedule
            current_alpha = self.get_alpha_for_iteration(i)
            print(f"=== Iteration {i+1}/{self.num_iter} (alpha={current_alpha:.6f}) ===")
            
            # Add random noise for robustness
            with torch.no_grad():
                noise = torch.randn_like(base_image) * noise_std
                noisy_image = base_image.clone() + noise
                noisy_image = torch.clamp(noisy_image, 0, 1)
            
            # Apply the modified first crop to the full tensor structure
            full_noisy_tensor = model_inputs['pixel_values'].clone()
            
            # Replace first crop with noisy image that has gradients
            noisy_image = noisy_image.requires_grad_(True)
            
            # Create new set of inputs with our modified tensor
            noisy_inputs = {k: v.clone() for k, v in model_inputs.items()}
            
            # For multi-crop models (Llama 3.2 Vision):
            # We need to place our modified first crop back into the full structure
            if len(pixel_shape) >= 4 and pixel_shape[2] > 1:
                # Create a new tensor with same shape as original
                modified_pixel_values = full_noisy_tensor.clone()
                
                # Place the noisy image (with gradients) as the first crop
                modified_pixel_values[:, :, 0] = noisy_image
                
                # Now all 4 crops have the same base image but only first has gradients
                noisy_inputs['pixel_values'] = modified_pixel_values
            else:
                # Simpler case - direct replacement
                noisy_inputs['pixel_values'] = noisy_image
            
            # Reset gradients and prepare for loss accumulation
            self.model.zero_grad()
            total_loss = 0
            
            # Create a working copy of inputs for token prediction
            working_inputs = {k: v.clone() for k, v in noisy_inputs.items()}
            
            # Number of tokens to optimize (limited by target sequence length)
            n_tokens = min(self.token_lookahead, len(target_token_ids))
            
            # For each token position in the sequence (up to lookahead)
            for j in range(n_tokens):
                # Get target token for this position
                target_id = target_token_ids[j]
                
                # Forward pass using standard model call instead of generate
                outputs = self.model(**working_inputs)
                
                # Get logits for the last position
                position_logits = outputs.logits[0, -1]
                
                # Calculate loss for this token
                log_probs = torch.log_softmax(position_logits, dim=-1)
                token_loss = -log_probs[target_id]
                
                # Apply position weighting 
                position_weight = max(float(self.token_lookahead - j), 0.5)
                
                # Add weighted loss to total
                weighted_loss = token_loss * position_weight
                total_loss = total_loss + weighted_loss
                
                # Print probabilities for monitoring
                probs = torch.softmax(position_logits, dim=-1)
                print(f"  Position {j}: Target '{self.tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
                top_token_id = torch.argmax(position_logits).item()
                print(f"    Top predicted: '{self.tokenizer.decode([top_token_id])}' prob: {probs[top_token_id].item():.4f}")
                
                # Update working inputs for next token prediction if needed
                if j < n_tokens - 1:
                    self.update_working_inputs(working_inputs, target_id)
            
            # Backpropagate the total loss
            total_loss.backward()
            
            # Track best result so far
            first_token_prob = probs[target_token_ids[0]].item()
            if first_token_prob > best_prob:
                best_prob = first_token_prob
                if len(pixel_shape) >= 4 and pixel_shape[2] > 1:
                    # Save just the first crop
                    best_image = base_image.clone()
                else:
                    best_image = base_image.clone()
            
            # Apply gradients if available
            if noisy_image.grad is not None:
                # Update momentum
                momentum = 0.9 * momentum - current_alpha * torch.sign(noisy_image.grad)
                
                # Apply momentum update to base image
                base_image = base_image + momentum
                
                # Project back to epsilon ball
                perturbation = torch.clamp(base_image - original_base_image, -self.epsilon, self.epsilon)
                base_image = original_base_image + perturbation
                
                # Ensure values are valid
                with torch.no_grad():
                    base_image = torch.clamp(base_image, 0, 1)
                    
                    # Update all crops in the full tensor with the modified base image
                    if len(pixel_shape) >= 4 and pixel_shape[2] > 1:
                        full_tensor = model_inputs['pixel_values'].clone()
                        # Apply the same perturbation to all crops
                        for crop_idx in range(pixel_shape[2]):
                            full_tensor[:, :, crop_idx] = base_image.clone()
                        model_inputs['pixel_values'] = full_tensor
                    else:
                        model_inputs['pixel_values'] = base_image.clone()
                    
                    # Reduce noise over time
                    noise_std = max(0.001, noise_std * 0.98)
                    
                    # Prepare for next iteration
                    base_image = base_image.detach().requires_grad_(True)
            else:
                print("Warning: No gradient calculated.")
            
            print()
        
        # Use the best image found during optimization
        if best_prob > 0.0 and self.return_best:
            print(f"\nUsing best image with probability {best_prob:.4f}")
            base_image = best_image
        else:
            print(f"\nUsing final image")
        
        # Create the final tensor with shape [1, 3, H, W]
        if len(base_image.shape) > 4:
            final_image = base_image.squeeze(1)  # Remove extra dimensions
        else:
            final_image = base_image
            
        print(f"Final output tensor shape: {final_image.shape}")
        
        return final_image


@AttackRegistry.register("llava")
class LlavaAttack(BasePGDAttack):
    """PGD attack implementation for LLaVA models."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For LLaVA models we need to ensure we have the proper tokenizer reference
        # Make sure we're using processor.tokenizer for encoding/decoding
        if not hasattr(self.tokenizer, 'encode'):
            self.tokenizer = self.processor.tokenizer
    
    def process_inputs(self, prompt, image):
        """Process inputs for LLaVA models"""
        # Format as conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template with direct tokenization
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        
        # Process image and add to inputs
        pixel_values = self.processor.image_processor(image, return_tensors="pt").pixel_values
        inputs["pixel_values"] = pixel_values
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate_from_inputs(self, inputs, max_new_tokens=10):
        """Generate text from processed inputs for LLaVA models"""
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_text = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            
        return generated_text
    
    def update_working_inputs(self, working_inputs, target_id):
        """Update inputs for next token prediction for LLaVA models"""
        # Add the target token to the input sequence
        if 'input_ids' in working_inputs:
            new_token_tensor = torch.tensor([[target_id]], device=working_inputs['input_ids'].device)
            working_inputs['input_ids'] = torch.cat([working_inputs['input_ids'], new_token_tensor], dim=1)
            
            # Also update attention mask if present
            if 'attention_mask' in working_inputs:
                working_inputs['attention_mask'] = torch.cat(
                    [working_inputs['attention_mask'], 
                    torch.ones((1, 1), device=working_inputs['attention_mask'].device)], 
                    dim=1
                )
    
    def pgd_attack(self, prompt, image_path, target_sequence):
        """
        Override the PGD attack for LLaVA models to handle tokenization differently.
        """
        # Process the image and text prompt
        image = Image.open(image_path).convert("RGB")
        model_inputs = self.process_inputs(prompt, image)
        
        # Set up for gradient tracking
        if 'pixel_values' not in model_inputs:
            raise ValueError("LLaVA inputs must contain 'pixel_values'")
        
        processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
        model_inputs['pixel_values'] = processed_image
        original_image = processed_image.clone()
        
        # Convert target sequence to token IDs using the tokenizer from the processor
        target_token_ids = self.tokenizer.encode(target_sequence, add_special_tokens=False)
        print(f"Target tokens: {target_sequence} -> {target_token_ids}")
        
        # The rest of the method follows the base implementation
        # Initialize momentum for more stable gradients
        momentum = torch.zeros_like(processed_image.data)
        
        # Initialize noise standard deviation
        noise_std = 0.01
        
        # Track best result
        best_image = processed_image.clone()
        best_prob = 0.0
        
        # Print scheduler information
        print(f"Using {self.scheduler_type} alpha scheduler with warmup ratio {self.warmup_ratio}")
        print(f"Alpha range: {self.alpha_min:.6f} to {self.alpha_max:.6f}")
        
        for i in range(self.num_iter):
            # Calculate current alpha based on schedule
            current_alpha = self.get_alpha_for_iteration(i)
            
            print(f"=== Iteration {i+1}/{self.num_iter} (alpha={current_alpha:.6f}) ===")

            # Add random noise to improve robustness against quantization
            with torch.no_grad():
                noise = torch.randn_like(processed_image) * noise_std
                noisy_image = processed_image.clone() + noise
                noisy_image = torch.clamp(noisy_image, 0, 1)
            
            # Create new inputs with noisy image
            noisy_inputs = {k: v.clone() for k, v in model_inputs.items()}
            noisy_inputs['pixel_values'] = noisy_image.requires_grad_(True)
            
            # Accumulate gradients over multiple tokens
            self.model.zero_grad()
            total_loss = 0
            
            # Create a working copy of inputs that we'll modify for sequential prediction
            working_inputs = {k: v.clone() for k, v in noisy_inputs.items()}
            
            # Number of tokens to optimize (limited by target sequence length)
            n_tokens = min(self.token_lookahead, len(target_token_ids))
            
            # For each token position in the sequence (up to lookahead)
            for j in range(n_tokens):
                # Get target token for this position
                target_id = target_token_ids[j]
                
                # Forward pass to get logits
                outputs = self.model.generate.__wrapped__(self.model, **working_inputs, return_dict_in_generate=True, output_scores=True, output_logits=True)
                
                # Get logits for the first predicted position (add 1 for space generated at end of "ASSISTANT:" output)
                position_logits = outputs['scores'][j+1][0]
                
                # Calculate loss for this token
                log_probs = torch.log_softmax(position_logits, dim=-1)
                token_loss = -log_probs[target_id]  # Negative log likelihood

                # Position weighting - focus strongly on first token with linear decay for subsequent tokens
                # First token gets weight=token_lookahead, second gets token_lookahead-1, etc.
                position_weight = max(float(self.token_lookahead - j), 0.5)  # Ensure minimum weight of 0.5
                
                # Add weighted loss to total
                weighted_loss = token_loss * position_weight
                total_loss = total_loss + weighted_loss
                
                # Print probabilities for monitoring
                probs = torch.softmax(position_logits, dim=-1)
                print(f"  Position {j}: Target '{self.tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
                top_token_id = torch.argmax(position_logits).item()
                print(f"    Top predicted: '{self.tokenizer.decode([top_token_id])}' prob: {probs[top_token_id].item():.4f}")
                
                # For next token prediction, append this token to input
                # Skip this step for the last token we're optimizing
                if j < n_tokens - 1:
                    # Update working inputs for next token prediction
                    self.update_working_inputs(working_inputs, target_id)
            
            # Backpropagate the total loss
            total_loss.backward()
            
            # Track first token probability for best result
            first_token_prob = probs[target_token_ids[0]].item()
            if first_token_prob > best_prob:
                best_prob = first_token_prob
                best_image = processed_image.clone()

            # PGD update with momentum
            if noisy_inputs['pixel_values'].grad is not None:
                # Update momentum term (helps stabilize optimization)
                momentum = 0.9 * momentum - current_alpha * torch.sign(noisy_inputs['pixel_values'].grad)
                
                # Apply momentum update
                processed_image = processed_image + momentum
                
                # Project back to epsilon ball
                perturbation = torch.clamp(processed_image - original_image, -self.epsilon, self.epsilon)
                processed_image = original_image + perturbation
                
                # Quantize directly to valid 8-bit values (multiples of 1/255)
                with torch.no_grad():
                    # Round to nearest 8-bit pixel values
                    # processed_image = torch.round(processed_image * 255) / 255
                    
                    # Ensure values stay in valid range after rounding
                    processed_image = torch.clamp(processed_image, 0, 1)
                    
                    # Since we're using discrete pixel values, small noise is still useful
                    # but we can use a smaller fixed value
                    noise_std = 0.003
                    
                    # Update model inputs with properly quantized image
                    model_inputs['pixel_values'] = processed_image.clone()
                    
                    # Prepare for next iteration
                    processed_image = processed_image.detach().requires_grad_(True)
            else:
                print("Warning: No gradient calculated.")
            
            print()

        # Check if the original image is equal to the processed image
        is_equal = torch.equal(original_image, processed_image)
        print(f"Original image is equal to processed image: {is_equal}")
        
        # Use the best image found during optimization
        if best_prob > 0.0 and self.return_best:
            print(f"\nUsing best image with probability {best_prob:.4f}")
            processed_image = best_image
        else:
            print(f"\nUsing final image")
            
        # The tensor retains the exact values without quantization
        tensor_image = processed_image.clone()
                
        return tensor_image


def pgd_attack(model_id, prompt, image_path, target_sequence, 
               epsilon=0.02, alpha_max=0.01, alpha_min=0.0001, num_iter=10, 
               token_lookahead=3, warmup_ratio=0.1, scheduler_type="cosine", return_best=True):
    """
    Factory function that creates and executes the appropriate PGD attack
    based on the model type.
    
    Args:
        model_id: HuggingFace model identifier
        prompt: Text prompt to use with the image
        image_path: Path to the input image
        target_sequence: Target text sequence to generate
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha_max: Maximum step size for PGD
        alpha_min: Minimum step size for PGD
        num_iter: Number of PGD iterations
        token_lookahead: Number of tokens ahead to optimize for
        warmup_ratio: Proportion of iterations to use for warming up alpha
        scheduler_type: Type of scheduler to use (linear, cosine, polynomial)
        
    Returns:
        PyTorch tensor containing the optimized image
    """
    # Get the appropriate attack class for this model
    attack_class = AttackRegistry.get_attack_function(model_id)
    
    # Create an instance with the specified parameters
    attack = attack_class(
        model_id=model_id,
        epsilon=epsilon,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        num_iter=num_iter,
        token_lookahead=token_lookahead,
        warmup_ratio=warmup_ratio,
        scheduler_type=scheduler_type,
        return_best=return_best
    )
    
    return attack.pgd_attack(prompt, image_path, target_sequence)