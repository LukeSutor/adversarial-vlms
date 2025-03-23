import os
from dotenv import load_dotenv
import torch
from PIL import Image
from huggingface_hub import login
import sys
# Append the current file directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from inference import model_manager, set_models_directory, run_inference, Models
from attacks import pgd_attack, get_token_id, get_generation
from utils import (load_image_tensor, convert_tensor_to_pil, save_attack_results, 
                  resize_images_with_padding, update_image_data_json, get_clean_filename)

def attack(input_image_path, model_id, target_sequence, prompt, epsilon=0.1, 
           alpha_max=0.01, alpha_min=0.0001, num_iter=200, warmup_ratio=0.2,
           scheduler_type="cosine"):
    """
    Perform an adversarial attack on a specified image using the given model.
    
    Args:
        input_image_path: Path to the clean image to attack
        model_id: Model to use for the attack
        target_sequence: Target text for the model to generate
        epsilon: Maximum perturbation magnitude
        alpha_max: Maximum step size for PGD
        alpha_min: Minimum step size for PGD
        num_iter: Number of PGD iterations
        warmup_ratio: Proportion of iterations to use for warming up alpha
        scheduler_type: Type of scheduler to use (linear, cosine, polynomial)
        prompt: Text prompt for the model
    """
    # Get the local path to the images
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    attack_dir = os.path.join(image_dir, "attack")
    clean_dir = os.path.join(image_dir, "clean")
    
    # Make sure input image exists
    if not os.path.exists(input_image_path):
        print(f"Error: Input image {input_image_path} does not exist.")
        return

    # Get clean filename without extension for naming the attack file
    clean_filename = get_clean_filename(input_image_path)
    
    # Save attacked images to attack directory with model info in filename
    attack_image_base = os.path.join(attack_dir, clean_filename)

    print(f"Performing attack on model: {model_id}...")
    print(f"Input image: {input_image_path}")
    print(f"Target sequence: {target_sequence}")
    
    tensor_image = pgd_attack(
        model_id=model_id, 
        prompt=prompt, 
        image_path=input_image_path, 
        target_sequence=target_sequence, 
        epsilon=epsilon, 
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        num_iter=num_iter,
        warmup_ratio=warmup_ratio,
        scheduler_type=scheduler_type
    )
    
    # Save both tensor and PNG versions using the utility function
    # Include model ID in the filename
    tensor_path, png_path = save_attack_results(tensor_image, attack_image_base, model_id)
    
    # Get the filename without extension for JSON entry
    attack_filename = get_clean_filename(tensor_path)
    
    # Run inference with the attacked image to get the model's response
    print("\nEvaluating attack results...")
    inference_result = run_inference(model_id, prompt, tensor_image)
    print(f"Attack result with {model_id.split('/')[-1]}:")
    print(inference_result)
    
    # Update the JSON file with this inference result
    update_image_data_json(attack_filename, model_id, inference_result)
    
    # Also run inference on the original image for comparison
    print("\nEvaluating original image for comparison...")
    original_result = get_generation(model_id, prompt, input_image_path)
    print(f"Original image response with {model_id.split('/')[-1]}:")
    print(original_result)
    
    return tensor_path, png_path, inference_result


def inference():
    # Hardcoded parameters
    tensor_name = "math_paligemma2-10b-mix-224"  # Updated to include model name
    model_id = Models.PALIGEMMA2_10B

    # Get the local path to the attack directory
    script_path = "/".join(__file__.split("/")[:-1])
    attack_dir = os.path.join(script_path, "../images/attack")

    # Construct the full path to the tensor file
    tensor_file_path = os.path.join(attack_dir, f"{tensor_name}.pt")

    # Check if the tensor file exists
    if not os.path.exists(tensor_file_path):
        print(f"Tensor file {tensor_file_path} does not exist.")
        return

    # Load the tensor
    tensor_image = load_image_tensor(tensor_file_path)

    # Run inference on the loaded tensor
    prompt = "Answer the question."
    inference_result = run_inference(model_id, prompt, tensor_image)
    print("\nInference result:", inference_result)
    
    # Update the JSON file with this inference result
    update_image_data_json(tensor_name, model_id, inference_result)

if __name__ == "__main__":
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])
    
    # Get the path to the clean images
    script_path = "/".join(__file__.split("/")[:-1])
    clean_dir = os.path.join(script_path, "../images/clean")
    
    # Specify the image file and model to use
    input_image = os.path.join(clean_dir, "math.png")
    model = Models.PALIGEMMA2_3B
    
    # Call the attack function with the specified image and model
    tensor_path, png_path, result = attack(
        input_image,
        model,
        "hello",
        "Answer the question.",
        num_iter=200  # Reduced for faster execution
    )
    
    print(f"\nAttack completed successfully.")
    print(f"Tensor saved to: {tensor_path}")
    print(f"PNG image saved to: {png_path}")
    print(f"Attack result: {result[:100]}...")  # Show first 100 chars of result