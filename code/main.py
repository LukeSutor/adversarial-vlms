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
from utils import load_image_tensor, convert_tensor_to_pil, save_attack_results, resize_images_with_padding

def attack():
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])

    # Select model to use
    model_id = Models.PALIGEMMA2_10B

    # Get the local path to the images
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    attack_dir = os.path.join(image_dir, "attack")
    clean_dir = os.path.join(image_dir, "clean")

    # Create directories if they don't exist
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # Define the attack parameters
    prompt = "Answer the question."
    target_sequence = "hello"
    input_image_path = os.path.join(clean_dir, "math_question.png")
    
    # Save attacked images to attack directory
    attack_image_base = os.path.join(attack_dir, "math")

    # Determine whether to run a new attack or use existing results
    attack = True

    if attack:
        print(f"Performing attack on model: {model_id}...")
        tensor_image = pgd_attack(
            model_id=model_id, 
            prompt=prompt, 
            image_path=input_image_path, 
            target_sequence=target_sequence, 
            epsilon=0.1, 
            alpha_max=0.01,  # Maximum step size 
            alpha_min=0.0001,  # Minimum step size
            num_iter=200,
            warmup_ratio=0.2,  # 20% of iterations for warmup
            scheduler_type="cosine"  # Options: linear, cosine, polynomial
        )
        
        # Save both tensor and PNG versions using the utility function
        tensor_path, png_path = save_attack_results(tensor_image, attack_image_base)
        
        # Optional: Run inference directly with the tensor we just created
        print("\nImmediate evaluation with tensor:")
        tensor_result = run_inference(model_id, prompt, tensor_image)
        print("Tensor-based answer:", tensor_result)

    # Evaluate the original image
    print("\nOriginal answer:", get_generation(model_id, prompt, input_image_path))
    
    # Check if attack files exist and evaluate them
    attack_tensor_file = f"{attack_image_base}.pt"
    if os.path.exists(attack_tensor_file):
        # Load and evaluate the tensor file
        tensor_image = load_image_tensor(attack_tensor_file)
        print("\nAttack answer (from tensor file):", run_inference(model_id, prompt, tensor_image))


def inference():
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])

    # Hardcoded parameters
    tensor_name = "math"
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
    print("\nInference result:", run_inference(model_id, prompt, tensor_image))

if __name__ == "__main__":
    attack()