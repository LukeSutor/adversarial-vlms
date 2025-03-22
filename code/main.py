import os
from dotenv import load_dotenv
import torch
from PIL import Image
from huggingface_hub import login

# Import local modules
from inference import model_manager, set_models_directory, run_inference, Models
from attacks import pgd_attack, get_token_id, get_generation
from utils import save_image, load_image_tensor

def main():
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])

    # Select model to use
    model_id = Models.PALIGEMMA2_3B

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
    attack_image_file = os.path.join(clean_dir, "math.png")

    # Determine whether to run a new attack or use existing results
    attack = True

    if attack:
        print(f"Performing attack on model: {model_id}...")
        perturbed_image, tensor_image = pgd_attack(
            model_id=model_id, 
            prompt=prompt, 
            image_path=input_image_path, 
            target_sequence=target_sequence, 
            epsilon=0.1, 
            alpha_max=0.01,  # Maximum step size 
            alpha_min=0.0001,  # Minimum step size
            num_iter=200,
            warmup_ratio=0.1,  # 10% of iterations for warmup
            scheduler_type="cosine"  # Options: linear, cosine, polynomial
        )
        
        # Save both PNG and tensor versions
        save_image(perturbed_image, attack_image_file, save_tensor=True)
        
        # Example of loading the saved tensor
        tensor_path = os.path.splitext(attack_image_file)[0] + '.pt'
        if os.path.exists(tensor_path):
            loaded_tensor = load_image_tensor(tensor_path)
            print(f"Successfully loaded tensor with shape: {loaded_tensor.shape}")
            # Evaluate the results based on the tensor as well as the PIL image
            print("\nEvaluating results based on the tensor:")
            tensor_result = get_generation(model_id, prompt, tensor_image)
            print("Tensor-based answer:", tensor_result)

    # Evaluate the original and attack image results
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


if __name__ == "__main__":
    main()
