from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, PaliGemmaProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import numpy as np
# Import the ModelManager from inference.py
from inference import ModelManager, set_models_directory, run_inference, Models


# Default model cache directory
DEFAULT_CACHE_DIR = "/blue/rcstudents/luke.sutor/adversarial-vlms/models/"

# Create or reuse the model manager
model_manager = ModelManager(cache_dir=DEFAULT_CACHE_DIR)


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
        elif "qwen2-vl" in model_id.lower():
            return cls._registry.get("qwen", cls._registry.get("default"))
        
        # Fall back to default
        return cls._registry.get("default")


# ==============================================
# Base Attack Implementation
# ==============================================

class BasePGDAttack:
    """Base class for PGD attack implementations."""
    
    def __init__(self, model_id, epsilon=0.02, alpha=0.01, num_iter=10, token_lookahead=3):
        self.model_id = model_id
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.token_lookahead = token_lookahead
        
        # Load model, tokenizer, and processor via model manager
        self.model, self.tokenizer, self.processor = model_manager.get_model(model_id)
    
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
        
        for i in range(self.num_iter):
            print(f"=== Iteration {i+1}/{self.num_iter} ===")

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
                momentum = 0.9 * momentum - self.alpha * torch.sign(noisy_inputs['pixel_values'].grad)
                
                # Apply momentum update
                processed_image = processed_image + momentum
                
                # Project back to epsilon ball
                perturbation = torch.clamp(processed_image - original_image, -self.epsilon, self.epsilon)
                processed_image = original_image + perturbation
                
                # Quantize directly to valid 8-bit values (multiples of 1/255)
                with torch.no_grad():
                    # Round to nearest 8-bit pixel values
                    processed_image = torch.round(processed_image * 255) / 255
                    
                    # Ensure values stay in valid range after rounding
                    processed_image = torch.clamp(processed_image, 0, 1)
                    
                    # Since we're using discrete pixel values, small noise is still useful
                    # but we can use a smaller fixed value
                    noise_std = 0.003
                    
                    # Update model inputs with properly quantized image
                    model_inputs['pixel_values'] = processed_image.clone()
                    
                    # Prepare for next iteration
                    processed_image = processed_image.detach().requires_grad_(True)
                    
                print(f"  Using quantized pixel values (1/255 steps) with noise std: {noise_std:.5f}")
            else:
                print("Warning: No gradient calculated.")
            
            print()

        # Use the best image found during optimization
        if best_prob > 0.0:
            print(f"\nUsing best image with probability {best_prob:.4f}")
            processed_image = best_image

        # Process the final perturbed image
        perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        # Final image with simulated quantization (important for robustness)
        perturbed_image = np.clip(perturbed_image * 255, 0, 255).round() / 255

        # Verify before saving (using processed image)
        print("\nVerifying attack BEFORE saving image...")
        with torch.no_grad():
            verify_inputs = {k: v.clone() for k, v in model_inputs.items()}
            verify_inputs['pixel_values'] = processed_image.clone()
            direct_result = self.generate_from_inputs(verify_inputs)
            print(f"Direct model output (before save): {direct_result}")
        
        return perturbed_image


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


def pgd_attack(model_id, prompt, image_path, target_sequence, 
               epsilon=0.02, alpha=0.01, num_iter=10, token_lookahead=3):
    """
    Factory function that creates and executes the appropriate PGD attack
    based on the model type.
    
    Args:
        model_id: HuggingFace model identifier
        prompt: Text prompt to use with the image
        image_path: Path to the input image
        target_sequence: Target text sequence to generate
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for PGD
        num_iter: Number of PGD iterations
        token_lookahead: Number of tokens ahead to optimize for
        
    Returns:
        Perturbed image as a numpy array
    """
    # Get the appropriate attack class for this model
    attack_class = AttackRegistry.get_attack_function(model_id)
    
    # Create an instance with the specified parameters
    attack = attack_class(
        model_id=model_id,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        token_lookahead=token_lookahead
    )
    
    # Run the attack
    return attack.pgd_attack(prompt, image_path, target_sequence)


def save_image(perturbed_image, output_path="perturbed_image.png"):
    """Save perturbed image with special handling for robustness."""
    # Convert the perturbed image to 8-bit format
    perturbed_image = (perturbed_image * 255).astype(np.uint8)
    
    # Create PIL image from array
    image = Image.fromarray(perturbed_image)
    
    # Save with no compression to preserve details
    image.save(output_path, format='PNG', compress_level=0, optimize=False)
    print(f"Saved attack image to {output_path}")


if __name__ == "__main__":
    load_dotenv()
    login(os.environ['HF_KEY'])
    # Available models:
    # model_id = Models.QWEN2_VL_2B
    # model_id = Models.PALIGEMMA_3B
    # model_id = Models.PALIGEMMA2_3B
    model_id = Models.PALIGEMMA2_10B
    # model_id = Models.PALIGEMMA2_28B
    # model_id = Models.PALIGEMMA2_3B

    # Get the local path to the images
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    attack_dir = os.path.join(image_dir, "attack")
    clean_dir = os.path.join(image_dir, "clean")

    prompt = "Answer the question."
    target_sequence = "hello"
    input_image_path = os.path.join(clean_dir, "math_question.png")
    attack_image_file = os.path.join(clean_dir, "math.png")

    attack = True

    if attack:
        print(f"Performing attack on model: {model_id}...")
        image = pgd_attack(
            model_id=model_id, 
            prompt=prompt, 
            image_path=input_image_path, 
            target_sequence=target_sequence, 
            epsilon=0.1, 
            alpha=0.01, 
            num_iter=500
        )
        save_image(image, attack_image_file)

    try:
        attack_image = Image.open(attack_image_file).convert("RGB")
        has_attack_image = True
    except:
        print(f"Attack image not found: {attack_image_file}")
        has_attack_image = False

    # Evaluate original and attack image results
    print("\nOriginal answer:", get_generation(model_id, prompt, input_image_path))
    if has_attack_image:
        print("\nAttack answer:", get_generation(model_id, prompt, attack_image_file))