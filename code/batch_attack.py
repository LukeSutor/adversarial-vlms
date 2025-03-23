import os
import json
import sys
from dotenv import load_dotenv
from huggingface_hub import login
import torch

# Ensure the current directory is in the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import local modules
from inference import model_manager, run_inference, Models
from main import attack
from utils import get_clean_filename

def process_image_batch():
    """
    Process all images from image_data.json and run attacks using their target answers.
    """
    # Setup environment and authentication
    load_dotenv()
    login(os.environ['HF_KEY'])
    
    # Configure paths
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    clean_dir = os.path.join(image_dir, "clean")
    attack_dir = os.path.join(image_dir, "attack")
    json_path = os.path.join(image_dir, "image_data.json")
    
    # Create attack directory if it doesn't exist
    os.makedirs(attack_dir, exist_ok=True)
    
    # Select model to use for all attacks
    model = Models.PALIGEMMA2_3B
    print(f"Using model: {model}")
    
    # Load image data
    try:
        with open(json_path, 'r') as file:
            image_data = json.load(file)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    
    # Check if clean images info exists
    if "clean" not in image_data or not image_data["clean"]:
        print("No clean images found in image_data.json")
        return
    
    # Set universal parameters
    prompt = "Answer the question."
    attack_params = {
        "epsilon": 0.1,
        "alpha_max": 0.01,
        "alpha_min": 0.0001,
        "num_iter": 200,
        "warmup_ratio": 0.2,
        "scheduler_type": "cosine"
    }
    
    # Process each image
    print(f"Found {len(image_data['clean'])} images to process")
    
    for i, image_info in enumerate(image_data["clean"]):
        image_name = image_info["name"]
        target_answer = image_info["target_answer"]
        
        print(f"\n[{i+1}/{len(image_data['clean'])}] Processing {image_name}")
        print(f"  Target answer: {target_answer}")
        
        # Full path to the clean image
        input_image_path = os.path.join(clean_dir, image_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(input_image_path):
            print(f"  WARNING: Image file {input_image_path} not found. Skipping.")
            continue
        
        try:
            # Run attack with parameters from the JSON file
            tensor_path, png_path, result = attack(
                input_image_path=input_image_path,
                model_id=model,
                target_sequence=target_answer,
                prompt=prompt,
                **attack_params
            )
            
            print(f"  Attack complete - Output: {result}")
            print(f"  Files saved: {tensor_path}, {png_path}")
            
        except Exception as e:
            print(f"  ERROR processing {image_name}: {e}")
    
    print("\nAll attacks completed!")

if __name__ == "__main__":
    # Run the batch processing
    process_image_batch()
