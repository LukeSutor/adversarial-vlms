from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import numpy as np


def get_token_id(tokenizer, string):
    # Gets the token ID of the specified string
    tokens = tokenizer.encode(string, add_special_tokens=False)
    if len(tokens) == 0:
        raise ValueError(f"The string '{string}' could not be tokenized.")
    return tokens[0]


def get_generation(processor, model, prompt, image_path):
    # Returns the generated text of passing an image through the model
    if model.name_or_path == "Qwen/Qwen2-VL-2B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text
    else:
        image = Image.open(image_path).convert("RGB")

        model_input = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_input["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**model_input, max_new_tokens=100, do_sample=False)
            final_generation = generation[0][input_len:]
            decoded_original = processor.decode(final_generation, skip_special_tokens=True)

        return decoded_original


def pgd_attack_sequence_advanced(model, tokenizer, processor, prompt, image_path, 
                               target_sequence, epsilon=0.02, alpha=0.01, num_iter=10,
                               token_lookahead=3):
    """
    Advanced PGD attack that optimizes an image to make a vision-language model generate
    a target sequence by iteratively optimizing for each token in the sequence.
    
    Args:
        model: The vision-language model
        tokenizer: Tokenizer for the model
        processor: Processor for the model
        prompt: Text prompt to use with the image
        image_path: Path to the input image
        target_sequence: Target text sequence to generate
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for PGD
        num_iter: Number of PGD iterations
        token_lookahead: Number of tokens ahead to optimize for
    """
    # Process the image and text prompt
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Set up for gradient tracking
    processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
    model_inputs['pixel_values'] = processed_image
    original_image = processed_image.clone()
    
    # Convert target sequence to token IDs
    target_token_ids = tokenizer.encode(target_sequence, add_special_tokens=False)
    print(f"Target tokens: {[tokenizer.decode([tid]) for tid in target_token_ids]}")
    
    # Initialize momentum for more stable gradients
    momentum = torch.zeros_like(processed_image.data)

    # Initialize noise standard deviation
    noise_std = 0.01
    
    # Track best result
    best_image = processed_image.clone()
    best_prob = 0.0
    
    for i in range(num_iter):
        print(f"=== Iteration {i+1}/{num_iter} ===")

        # Add random noise to improve robustness against quantization
        with torch.no_grad():
            noise = torch.randn_like(processed_image) * noise_std
            noisy_image = processed_image.clone() + noise
            noisy_image = torch.clamp(noisy_image, 0, 1)
        
        # Create new inputs with noisy image
        noisy_inputs = {k: v.clone() for k, v in model_inputs.items()}
        noisy_inputs['pixel_values'] = noisy_image.requires_grad_(True)
        
        # Accumulate gradients over multiple tokens
        model.zero_grad()
        total_loss = 0
        
        # Create a working copy of inputs that we'll modify for sequential prediction
        working_inputs = {k: v.clone() for k, v in noisy_inputs.items()}
        
        # Number of tokens to optimize (limited by target sequence length)
        n_tokens = min(token_lookahead, len(target_token_ids))
        
        # For each token position in the sequence (up to lookahead)
        for j in range(n_tokens):
            # Get target token for this position
            target_id = target_token_ids[j]
            
            # Forward pass to get logits
            outputs = model(**working_inputs)
            
            # Get logits for the last position (predicting next token)
            position_logits = outputs.logits[0, -1]
            
            # Calculate loss for this token
            log_probs = torch.log_softmax(position_logits, dim=-1)
            token_loss = -log_probs[target_id]  # Negative log likelihood

            # Position weighting - focus strongly on first token with linear decay for subsequent tokens
            # First token gets weight=token_lookahead, second gets token_lookahead-1, etc.
            position_weight = max(float(token_lookahead - j), 0.5)  # Ensure minimum weight of 0.5
            
            # Add weighted loss to total
            weighted_loss = token_loss * position_weight
            total_loss = total_loss + weighted_loss
            
            # Print probabilities for monitoring
            probs = torch.softmax(position_logits, dim=-1)
            print(f"  Position {j}: Target '{tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
            top_token_id = torch.argmax(position_logits).item()
            print(f"    Top predicted: '{tokenizer.decode([top_token_id])}' prob: {probs[top_token_id].item():.4f}")
            
            # For next token prediction, append this token to input
            # Skip this step for the last token we're optimizing
            if j < n_tokens - 1:
                # Prepare for next token by adding current target token to the input sequence
                new_token_tensor = torch.tensor([[target_id]], device=working_inputs['input_ids'].device)
                working_inputs['input_ids'] = torch.cat([working_inputs['input_ids'], new_token_tensor], dim=1)
                
                # Also update attention mask if present
                if 'attention_mask' in working_inputs:
                    working_inputs['attention_mask'] = torch.cat(
                        [working_inputs['attention_mask'], 
                         torch.ones((1, 1), device=working_inputs['attention_mask'].device)], 
                        dim=1
                    )
        
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
            momentum = 0.9 * momentum - alpha * torch.sign(noisy_inputs['pixel_values'].grad)
            
            # Apply momentum update
            processed_image = processed_image + momentum
            
            # Project back to epsilon ball and valid image range
            perturbation = torch.clamp(processed_image - original_image, -epsilon, epsilon)
            processed_image = torch.clamp(original_image + perturbation, 0, 1).detach().requires_grad_(True)
            
            # Simulate quantization to estimate its effect
            quantized = (processed_image.clone().cpu() * 255).round() / 255
            quantization_diff = processed_image.cpu() - quantized

            # Save the processed image as the quantized version
            model_inputs['pixel_values'] = quantized
            
            # Update noise standard deviation based on quantization error
            noise_std = quantization_diff.std().item()
            print(f"  Updated noise std: {noise_std:.5f}")
        else:
            print("Warning: No gradient calculated.")
    
        print()

    # Use the best image found during optimization
    if best_prob > 0.0:
        print(f"\nUsing best image with probability {best_prob:.4f}")
        processed_image = best_image

    # Process the final perturbed image
    perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Verify before saving (using direct tensor)
    print("\nVerifying attack BEFORE saving image...")
    with torch.no_grad():
        verify_inputs = {k: v.clone() for k, v in model_inputs.items()}
        verify_inputs['pixel_values'] = processed_image
        direct_output = model.generate(**verify_inputs, max_new_tokens=10)
        direct_result = processor.decode(direct_output[0][len(verify_inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Direct model output (before save): {direct_result}")
    
    # Final image with simulated quantization (important for robustness)
    perturbed_image = np.clip(perturbed_image * 255, 0, 255).round() / 255
    
    return perturbed_image
    

def pgd_attack_sequence(model, tokenizer, processor, prompt, image_path, 
                        target_sequence, epsilon=0.02, alpha=0.01, num_iter=10):
    """
    PGD attack that optimizes an image to make a vision-language model generate a target sequence.
    """
    # Process the image and text prompt
    if model.name_or_path == "Qwen/Qwen2-VL-2B-Instruct":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(model.device)
    else:
        image = Image.open(image_path).convert("RGB")
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Set up for gradient tracking
    processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
    model_inputs['pixel_values'] = processed_image
    original_image = processed_image.clone()
    
    # Convert target sequence to token IDs
    target_token_ids = tokenizer.encode(target_sequence, add_special_tokens=False)
    
    # Get information about the input sequence length
    input_ids = model_inputs['input_ids'].clone()
    
    for i in range(num_iter):
        print(f"Iteration: {i}")
        
        # Forward pass to get logits
        outputs = model(**model_inputs).logits
        
        # Focus on the last position to predict the next token
        last_position_logits = outputs[0, -1]  # Shape: [vocab_size]
        
        # Calculate loss for the first token in target sequence
        target_id = target_token_ids[0]
        log_probs = torch.log_softmax(last_position_logits, dim=-1)
        loss = -log_probs[target_id]  # Negative log likelihood
        
        # Print probabilities for monitoring
        probs = torch.softmax(last_position_logits, dim=-1)
        print(f"Target token '{tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
        top_token_id = torch.argmax(last_position_logits).item()
        print(f"Top predicted: '{tokenizer.decode([top_token_id])}' prob: {probs[top_token_id].item():.4f}")
        
        # Backpropagate and update the image
        model.zero_grad()
        loss.backward()
        
        # PGD update
        if processed_image.grad is not None:
            perturbation = -alpha * torch.sign(processed_image.grad)
            processed_image = processed_image + perturbation
            perturbation = torch.clamp(processed_image - original_image, -epsilon, epsilon)
            processed_image = torch.clamp(original_image + perturbation, 0, 1).detach().requires_grad_(True)
            model_inputs['pixel_values'] = processed_image
        else:
            print("Warning: No gradient calculated. Check if model supports backpropagation.")

    # Process the final perturbed image
    perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    # Get final generation for verification (without gradients)
    with torch.no_grad():
        verify_inputs = model_inputs.copy()
        verify_inputs['pixel_values'] = processed_image
        
        # Generate text to verify attack success
        if model.name_or_path == "Qwen/Qwen2-VL-2B-Instruct":
            generated_ids = model.generate(**verify_inputs, max_new_tokens=len(target_token_ids) + 10)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(verify_inputs.input_ids, generated_ids)
            ]
            final_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            generated = model.generate(**verify_inputs, max_new_tokens=len(target_token_ids) + 10)
            final_output = processor.decode(
                generated[0][len(verify_inputs['input_ids'][0]):], skip_special_tokens=True
            )
        
        print(f"Target sequence: {target_sequence}")
        print(f"Generated output: {final_output}")
    
    return perturbed_image


def save_image(perturbed_image, output_path="perturbed_image.png"):
    """Save perturbed image with special handling for robustness."""
    # Convert the perturbed image to 8-bit format
    perturbed_image = (perturbed_image * 255).astype(np.uint8)
    
    # Create PIL image from array
    image = Image.fromarray(perturbed_image)
    
    # Save with minimal compression to preserve details
    image.save(output_path, format='PNG', compress_level=0)
    print(f"Saved attack image to {output_path}")


def load_model(model_id):
    # Loads a model preset based on id
    device = "cuda:0"
    dtype = torch.float16

    if model_id == "google/paligemma-3b-mix-224":
        return PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            revision="float16"
        ).eval()
    elif model_id == "Qwen/Qwen2-VL-2B-Instruct":
        return Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
    else:
        raise ValueError("Unknown model requested")


if __name__ == "__main__":
    # load_dotenv()
    # login(os.environ.get("HF_TOKEN"))

    # model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model_id = "google/paligemma-3b-mix-224"

    model = load_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = "Answer the question."
    target_sequence = "hello"
    image_path = "math_question.png"
    attack_image_file = "attack_image.png"

    attack = True

    if attack:
        print("Performing attack...")
        # image = pgd_attack_sequence(model, tokenizer, processor, prompt, image_path, target_sequence, epsilon=0.1, alpha=0.001, num_iter=100)
        image = pgd_attack_sequence_advanced(model, tokenizer, processor, prompt, image_path, target_sequence, epsilon=0.1, alpha=1e-4, num_iter=500)
        save_image(image, attack_image_file)

    try:
        attack_image = Image.open(attack_image_file).convert("RGB")
    except:
        attack_image = None

    # Evaluate original and attack image results
    print("Original answer:", get_generation(processor, model, prompt, image_path))
    if attack_image:
        print("Attack answer:", get_generation(processor, model, prompt, attack_image_file))