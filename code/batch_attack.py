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
    Updated to work with the new JSON structure.
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
    
    # Check if images exist in the new format
    if "images" not in image_data or not image_data["images"]:
        print("No images found in image_data.json")
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
    images_to_process = image_data["images"]
    print(f"Found {len(images_to_process)} images to process")
    
    for i, (image_name, image_info) in enumerate(images_to_process.items()):
        metadata = image_info.get("metadata", {})
        filename = metadata.get("filename", f"{image_name}.png")
        target_answer = metadata.get("target_answer", "")
        question = metadata.get("question", "")
        
        print(f"\n[{i+1}/{len(images_to_process)}] Processing {filename}")
        if question:
            print(f"  Question: {question}")
        print(f"  Target answer: {target_answer}")
        
        # Full path to the clean image
        input_image_path = os.path.join(clean_dir, filename)
        
        # Skip if image doesn't exist
        if not os.path.exists(input_image_path):
            print(f"  WARNING: Image file {input_image_path} not found. Skipping.")
            continue
        
        # Skip if no target answer is provided
        if not target_answer:
            print(f"  WARNING: No target answer provided for {filename}. Skipping.")
            continue
        
        try:
            # Run attack with parameters from the JSON file
            tensor_path, result = attack(
                input_image_path=input_image_path,
                model_id=model,
                target_sequence=target_answer,
                prompt=prompt,
                **attack_params
            )
            
            print(f"  Attack complete - Output: {result}")
            print(f"  File saved: {tensor_path}")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
    
    print("\nAll attacks completed!")

if __name__ == "__main__":
    # Run the batch processing
    process_image_batch()
