from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
from PIL import Image
from huggingface_hub import login
import torch
import numpy as np


def pgd_attack(model, processor, prompt, image, target_token_idx, original_target_class, new_target_class, epsilon=0.02, alpha=0.01, num_iter=10):
    # Process the image and text prompt
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # Clone the image tensor from the model_inputs and set requires_grad to True for the image tensor
    processed_image = model_inputs['pixel_values'].clone().detach().requires_grad_(True)
    
    # Replace the 'pixel_values' in model inputs with the one that has gradients enabled
    model_inputs['pixel_values'] = processed_image
 
    original_image = processed_image.clone()

    for i in range(num_iter):
        print("Iteration:", i)
        # Forward pass: get the logits from the model
        output = model(**model_inputs).logits  # Shape: [batch_size, seq_length, vocab_size]

        # Print probability for target class
        probabilities = torch.nn.functional.softmax(output, dim=-1)
        print("Original target class probability:", probabilities[0][target_token_idx][original_target_class].item())
        print("New target class probability:", probabilities[0][target_token_idx][new_target_class].item())
        print()

        # Select logits corresponding to the target token (e.g., second-to-last token)
        target_logits = output[0, target_token_idx]  # Shape: [vocab_size]

        # Calculate the loss:
        # - Minimize the logit of the original target class
        # - Maximize the logit of the new target class
        loss = -target_logits[original_target_class] + target_logits[new_target_class]

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

    # Convert the perturbed tensor back to an image if needed
    perturbed_image = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    return perturbed_image, final_logits


def save_image(perturbed_image, output_path="perturbed_image.png"):
    # Convert the perturbed image (NumPy array) back to an image
    perturbed_image = (perturbed_image * 255).astype(np.uint8)  # Scale back to [0, 255] range
    image = Image.fromarray(perturbed_image)

    image.save(output_path)


def get_token_id(tokenizer, string):
    # Gets the token ID of the specified string
    tokens = tokenizer.encode(string, add_special_tokens=False)
    if len(tokens) == 0:
        raise ValueError(f"The string '{string}' could not be tokenized.")
    return tokens[0]


if __name__ == "__main__":
    model_id = "google/paligemma-3b-mix-224"
    model_dir = "PaliGemma"
    device = "cuda:0"
    dtype = torch.float16

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        revision="float16"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    import pdb; pdb.set_trace()

    processor = AutoProcessor.from_pretrained(model_id)

    prompt = "What is the answer to this question? Back up your answer with an explanation"
    input_image_file = "C:/Users/Luke/Desktop/UF/Artificial_Intelligence_Scholars/year_2/question_images/input_image.png"
    attack_image_file = "C:/Users/Luke/Desktop/UF/Artificial_Intelligence_Scholars/year_2/question_images/attack_image.png"

    attack = False

    input_image = Image.open(input_image_file).convert("RGB")
    try:
        attack_image = Image.open(attack_image_file).convert("RGB")
    except:
        attack_image = None

    if attack:
        image, final_logits = pgd_attack(model, processor, prompt, input_image, -2, 235268, 235260, epsilon=0.5, alpha=0.01, num_iter=100)
        print(final_logits)
        save_image(image, "attack_image.png")
    else:
        model_inputs_original = processor(text=prompt, images=input_image, return_tensors="pt").to(model.device)
        input_len_original = model_inputs_original["input_ids"].shape[-1]

        if attack_image:
            model_inputs_attack = processor(text=prompt, images=attack_image, return_tensors="pt").to(model.device)
            input_len_attack = model_inputs_attack["input_ids"].shape[-1]

        with torch.inference_mode():
            # model_outputs = model(**model_inputs)
            # logits = model_outputs.logits

            # # Apply softmax to the logits along the last dimension (over the possible output classes)
            # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # # Get the highest probability class for each token
            # top_probabilities, top_classes = torch.max(probabilities, dim=-1)

            generation_original = model.generate(**model_inputs_original, max_new_tokens=100, do_sample=False)
            final_generation_original = generation_original[0][input_len_original:]
            decoded_original = processor.decode(final_generation_original, skip_special_tokens=True)

            if attack_image:
                generation_attack = model.generate(**model_inputs_attack, max_new_tokens=100, do_sample=False)
                final_generation_attack = generation_attack[0][input_len_attack:]
                decoded_attack = processor.decode(final_generation_attack, skip_special_tokens=True)

            print("Original image answer:", decoded_original)
            if attack_image:
                print("Attack image answer:", decoded_attack)
