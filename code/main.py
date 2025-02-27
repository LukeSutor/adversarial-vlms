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
    

def pgd_attack_sequence(model, tokenizer, processor, prompt, image_path, 
                        correct_sequence, target_sequence, 
                        epsilon=0.02, alpha=0.01, num_iter=10):

    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # Set up for gradient tracking
    processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
    model_inputs['pixel_values'] = processed_image
    original_image = processed_image.clone()
    
    # Convert target sequences to token IDs
    correct_token_ids = tokenizer.encode(correct_sequence, add_special_tokens=False)
    target_token_ids = tokenizer.encode(target_sequence, add_special_tokens=False)
    
    for i in range(num_iter):
        print(f"Iteration: {i}")
        # Forward pass to get logits
        outputs = model(**model_inputs).logits
        
        # Initialize loss
        loss = 0
        
        # Process for autoregressive sequence prediction
        # Method 1: Use the model's predicted tokens at each position
        input_ids = model_inputs['input_ids'].clone()
        seq_len = input_ids.shape[1]
        
        # Calculate loss across sequence positions (comparing correct vs target tokens)
        for j, (correct_id, target_id) in enumerate(zip(correct_token_ids, target_token_ids)):
            if seq_len + j >= outputs.shape[1]:
                break  # Don't exceed sequence length
                
            # Get probabilities at this position
            position_logits = outputs[0, seq_len + j - 1]  # -1 because we're predicting the next token
            probs = torch.nn.functional.softmax(position_logits, dim=-1)
            
            # Print probabilities for monitoring
            print(f"Position {j}: Correct token '{tokenizer.decode([correct_id])}' prob: {probs[correct_id].item():.4f}, "
                  f"Target token '{tokenizer.decode([target_id])}' prob: {probs[target_id].item():.4f}")
            
            # Add to loss (minimize correct, maximize target)
            loss = loss - position_logits[target_id] + position_logits[correct_id]
        
        # Alternative Method 2: Teacher forcing approach
        # This forces the model to predict based on the target sequence
        forced_inputs = model_inputs.copy()
        
        # For each position in the sequence:
        # 1. Run the model to get the next token prediction
        # 2. Calculate loss between that prediction and target token
        # 3. Update the input sequence with the target token
        # (Code for this approach would be more complex and model-specific)
        
        # Backpropagate and update the image
        model.zero_grad()
        loss.backward()
        
        # PGD update
        perturbation = alpha * torch.sign(processed_image.grad)
        processed_image = processed_image + perturbation
        perturbation = torch.clamp(processed_image - original_image, -epsilon, epsilon)
        processed_image = torch.clamp(original_image + perturbation, 0, 1).detach().requires_grad_(True)
        model_inputs['pixel_values'] = processed_image

    # Process the final perturbed image as in your existing code
    # [Your existing image processing code]
    perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    return perturbed_image, outputs


def pgd_attack(model, tokenizer, processor, prompt, image_path, target_token_idx, correct_answer, target_answer, epsilon=0.02, alpha=0.01, num_iter=10):
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

        # Process the image and text prompt
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # Clone the image tensor from the model_inputs and set requires_grad to True for the image tensor
    processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
    
    # Replace the 'pixel_values' in model inputs with the one that has gradients enabled
    model_inputs['pixel_values'] = processed_image

    original_image = processed_image.clone()

    # Get the token IDs of the incorrect and correct answer tokens
    original_classes = [get_token_id(tokenizer, answer) for answer in [correct_answer.lower(), correct_answer.upper()]]
    target_classes = [get_token_id(tokenizer, answer) for answer in [target_answer.lower(), target_answer.upper()]]

    for i in range(num_iter):
        print("Iteration:", i)
        # Forward pass: get the logits from the model
        output = model(**model_inputs).logits  # Shape: [batch_size, seq_length, vocab_size]

        # Print probability for target class
        probabilities = torch.nn.functional.softmax(output, dim=-1)
        print(f"Original class probability ({tokenizer.decode([original_classes[0]])}):", probabilities[0][target_token_idx][original_classes[0]].item())
        print(f"Target class probability ({tokenizer.decode([target_classes[0]])}):", probabilities[0][target_token_idx][target_classes[0]].item())
        print()

        # Select logits corresponding to the target token (e.g., second-to-last token)
        target_logits = output[0, target_token_idx]  # Shape: [vocab_size]

        # Calculate the loss:
        # - Minimize the logit of the original target class
        # - Maximize the logit of the new target class
        loss = -sum(target_logits[original_class] for original_class in original_classes) + sum(target_logits[target_class] for target_class in target_classes)

        model.zero_grad()
        loss.backward()

        # PGD update: Add perturbation in the direction of the gradient to the processed image
        perturbation = alpha * torch.sign(processed_image.grad)

        # Apply the perturbation and project it back to the valid epsilon-ball around the original image
        processed_image = processed_image + perturbation
        perturbation = torch.clamp(processed_image - original_image, -epsilon, epsilon)
        processed_image = torch.clamp(original_image + perturbation, 0, 1).detach().requires_grad_(True)

        # Update the model_inputs with the perturbed image
        model_inputs['pixel_values'] = processed_image

    # Return the perturbed image and the final logits for the target token
    final_logits = model(**model_inputs).logits[0, target_token_idx]

    # Ensure the processed_image is in the correct shape
    if model.name_or_path == "Qwen/Qwen2-VL-2B-Instruct":
        # Reshape to [1, 1, height, width] if it's a 2D tensor
        processed_image = processed_image.unsqueeze(0).unsqueeze(0)
    # elif processed_image.dim() == 3:
    #     # Reshape to [1, channels, height, width] if it's a 3D tensor
    #     processed_image = processed_image.unsqueeze(0)

    # Convert the perturbed tensor back to an image if needed
    perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Ensure the perturbed_image has the correct shape and data type
    if model.name_or_path == "Qwen/Qwen2-VL-2B-Instruct":
        # If the image has a single channel, convert it to 3 channels by repeating the channel
        perturbed_image = np.repeat(perturbed_image, 3, axis=-1)
        perturbed_image = (perturbed_image * 255).astype(np.uint8)

    return perturbed_image, final_logits


def save_image(perturbed_image, output_path="perturbed_image.png"):
    # Convert the perturbed image (NumPy array) back to an image
    perturbed_image = (perturbed_image * 255).astype(np.uint8)  # Scale back to [0, 255] range
    image = Image.fromarray(perturbed_image)

    image.save(output_path)


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
    load_dotenv()
    login(os.environ("HF_TOKEN"))

    # model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model_id = "google/paligemma-3b-mix-224"

    model = load_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = "What is the answer to this question? Answer with just the letter."
    image_path = "math_question.png"
    attack_image_file = "attack_image.png"

    attack = True

    if attack:
        image, final_logits = pgd_attack(model, tokenizer, processor, prompt, image_path, -2, "b", "d", epsilon=0.0002, alpha=0.0001, num_iter=10)
        print(final_logits)
        save_image(image, attack_image_file)

    try:
        attack_image = Image.open(attack_image_file).convert("RGB")
    except:
        attack_image = None

    # Evaluate original and attack image results
    print("Original answer:", get_generation(processor, model, prompt, image_path))
    if attack_image:
        print("Attack answer:", get_generation(processor, model, prompt, attack_image_file))

    if attack_image:
        model_inputs_attack = processor(text=prompt, images=attack_image, return_tensors="pt").to(model.device)
        input_len_attack = model_inputs_attack["input_ids"].shape[-1]