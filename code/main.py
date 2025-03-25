import os
from dotenv import load_dotenv
import torch
from PIL import Image
from huggingface_hub import login
import json
import sys
import time
# Append the current file directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from inference import model_manager, set_models_directory, run_inference, Models
from attacks import pgd_attack, get_token_id, get_generation
from utils import (load_image_tensor, convert_tensor_to_pil, save_attack_results, 
                  resize_images_with_padding, update_image_data_json, get_clean_filename, update_clean_image_output)

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
    
    # Save attacked images to attack directory (model subfolder will be created in save_attack_results)
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
    
    # Save only the tensor version (PT file) in model-specific subfolder
    tensor_path = save_attack_results(tensor_image, attack_image_base, model_id)
    
    # Get the filename without extension for JSON entry
    attack_filename = get_clean_filename(tensor_path)
    
    inference_result = None
    # Run inference with the attacked image to get the model's response
    print("\nEvaluating attack results...")
    inference_result = run_inference(model_id, prompt, tensor_image)
    print(f"Attack result with {model_id.split('/')[-1]}:")
    print(inference_result)
    
    # Collect attack parameters for JSON
    attack_params = {
        "epsilon": epsilon,
        "alpha": alpha_max,  # Save the maximum alpha
        "iterations": num_iter,
        "scheduler": scheduler_type,
        "warmup_ratio": warmup_ratio
    }
    
    # Update the JSON file with this inference result and attack parameters
    update_image_data_json(attack_filename, model_id, inference_result, attack_params)
    
    # Also run inference on the original image for comparison
    print("\nEvaluating original image for comparison...")
    original_result = get_generation(model_id, prompt, input_image_path)
    print(f"Original image response with {model_id.split('/')[-1]}:")
    print(original_result)
    
    return tensor_path, inference_result


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


def evaluate_clean_images(models, prompt="Answer the question."):
    """
    Evaluate all clean images with specified models and update image_data.json.
    Works with the new JSON structure where images are stored under the "images" key.
    Ensures that each image has the required structure fields.
    
    Args:
        models: List of model IDs to use for evaluation
        prompt: Text prompt to use with each image
    """
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])
    
    # Get paths
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    clean_dir = os.path.join(image_dir, "clean")
    json_path = os.path.join(image_dir, "image_data.json")
    
    # Load image data
    try:
        with open(json_path, 'r') as file:
            image_data = json.load(file)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    
    # Ensure images dict exists
    if "images" not in image_data or not image_data["images"]:
        print("No images found in image_data.json")
        return
    
    # Count how many images we'll process
    num_images = len(image_data["images"])
    print(f"Evaluating {num_images} clean images with {len(models)} models")
    
    # Initialize the structure for each image if missing
    for image_name, image_info in image_data["images"].items():
        filename = image_info.get("metadata", {}).get("filename", f"{image_name}.png")
        
        # Ensure clean field exists
        if "clean" not in image_info:
            image_info["clean"] = {
                "path": f"clean/{filename}",
                "model_outputs": {}
            }
        
        # Ensure attacks field exists
        if "attacks" not in image_info:
            image_info["attacks"] = {}
    
    # Save the updated structure back to the file
    try:
        with open(json_path, 'w') as file:
            json.dump(image_data, file, indent=2)
        print(f"Updated JSON file with required structure fields")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
    
    # Process each image with each model
    for i, (image_name, image_info) in enumerate(image_data["images"].items()):
        filename = image_info.get("metadata", {}).get("filename", f"{image_name}.png")
        image_path = os.path.join(clean_dir, filename)
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"WARNING: Image file {image_path} not found. Skipping.")
            continue
        
        print(f"\n[{i+1}/{num_images}] Evaluating {filename}")
        
        # Get the question for context
        question = image_info.get("metadata", {}).get("question", "")
        if question:
            print(f"  Question: {question}")
        
        # Evaluate with each model
        for model_id in models:
            model_name = model_id.split('/')[-1]
            print(f"  Using model: {model_name}")
            
            try:
                # Run inference
                result = run_inference(model_id, prompt, image_path)
                print(f"  Output: {result}")
                
                # Update JSON with results using the new structure
                update_clean_image_output(image_name, model_id, result)
                
            except Exception as e:
                print(f"  ERROR with model {model_name}: {str(e)}")
    
    print("\nAll evaluations completed!")


def evaluate_attack_images(models, prompt="Answer the question.", replace_existing=False):
    """
    Evaluate all attack images with specified models and update image_data.json.
    Works with the new hierarchical data structure where attack images are stored
    in model-specific subfolders.
    
    Args:
        models: List of model IDs to use for evaluation
        prompt: Text prompt to use with each image
        replace_existing: If False, skip evaluations for model-attack combinations 
                          that already have results
    """
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])
    
    # Get paths
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    attack_dir = os.path.join(image_dir, "attack")
    json_path = os.path.join(image_dir, "image_data.json")
    
    # Load image data
    try:
        with open(json_path, 'r') as file:
            image_data = json.load(file)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    
    # Ensure images dict exists
    if "images" not in image_data:
        print("No images found in image_data.json")
        return
    
    # Count how many attack images we'll process
    attack_count = 0
    total_images = 0
    for image_name, image_info in image_data["images"].items():
        if "attacks" in image_info and image_info["attacks"]:
            attack_count += len(image_info["attacks"])
            total_images += 1
    
    if attack_count == 0:
        print("No attack images found in image_data.json")
        return
    
    print(f"Found {attack_count} attack images across {total_images} base images")
    print(f"Evaluating with {len(models)} models")
    
    # Process each image with each model
    processed_count = 0
    
    for image_name, image_info in image_data["images"].items():
        if "attacks" not in image_info or not image_info["attacks"]:
            continue
            
        print(f"\nProcessing attacks for image: {image_name}")
        
        # Get the question for context
        question = image_info.get("metadata", {}).get("question", "")
        if question:
            print(f"  Question: {question}")
        
        # Process each attack for this image
        for attack_model_name, attack_info in image_info["attacks"].items():
            processed_count += 1
            print(f"\n  [{processed_count}/{attack_count}] Evaluating {attack_model_name} attack")
            
            # Get the path to the attack tensor
            attack_path = attack_info.get("metadata", {}).get("path").lower()
            if not attack_path:
                print(f"    Missing path for {attack_model_name} attack, skipping")
                continue
                
            tensor_path = os.path.join(image_dir, attack_path)
            if not os.path.exists(tensor_path):
                print(f"    Attack file not found: {tensor_path}, skipping")
                continue
                
            # Load the tensor
            try:
                tensor_image = load_image_tensor(tensor_path)
            except Exception as e:
                print(f"    Error loading tensor: {e}, skipping")
                continue
                
            # Run inference with each model
            for model_id in models:
                model_name = model_id.split('/')[-1]
                print(f"    Using model: {model_name}")
                
                # Check if we should skip this model due to existing output
                if not replace_existing and model_id in attack_info.get("model_outputs", {}):
                    print(f"      Skipping - Output already exists and replace_existing=False")
                    continue
                    
                try:
                    # Run inference
                    result = run_inference(model_id, prompt, tensor_image)
                    print(f"      Output: {result}")
                    
                    # Create timestamp for tracking when inference was run
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Update the model outputs for this attack
                    if "model_outputs" not in attack_info:
                        attack_info["model_outputs"] = {}
                        
                    # Add or update the model result
                    attack_info["model_outputs"][model_id] = {
                        "output": result,
                        "timestamp": timestamp
                    }
                    
                except Exception as e:
                    print(f"      ERROR with model {model_name}: {str(e)}")
    
    # Write updated data back to file
    try:
        with open(json_path, 'w') as file:
            json.dump(image_data, file, indent=2)
        print(f"\nUpdated {json_path} with attack evaluation results")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
    
    print("\nAll evaluations completed!")


if __name__ == "__main__":
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])

    # attack("/blue/rcstudents/luke.sutor/adversarial-vlms/images/clean/boiling.png", Models.LLAMA_3_2_11B_VISION, "A", "Answer the question")
    
    # Example of evaluating attack images with multiple models
    evaluate_attack_images([Models.PALIGEMMA2_3B])

    # Alternatively, run the clean image evaluation
    # evaluate_clean_images([Models.PALIGEMMA2_3B])