import torch
import numpy as np
from PIL import Image
import os
import json
import time


def load_image_tensor(input_path, device='cpu'):
    """
    Load an image tensor from a PyTorch tensor file (.pt).
    
    Args:
        input_path: Path to the tensor file
        device: Device to load the tensor to ('cpu', 'cuda', etc.)
        
    Returns:
        PyTorch tensor containing the image data
    """
    tensor_image = torch.load(input_path, map_location=device)
    print(f"Loaded tensor image from {input_path}, shape: {tensor_image.shape}")
    return tensor_image


def convert_tensor_to_pil(tensor_image):
    """
    Convert a PyTorch tensor image to a PIL Image.
    
    Args:
        tensor_image: PyTorch tensor with shape [C, H, W] or [1, C, H, W]
        
    Returns:
        PIL Image object
    """
    # Make sure tensor is on CPU and detached from computation graph
    if tensor_image.requires_grad:
        tensor_image = tensor_image.detach()
    
    if tensor_image.device.type != 'cpu':
        tensor_image = tensor_image.cpu()
    
    # Remove batch dimension if present
    if tensor_image.dim() == 4:
        tensor_image = tensor_image.squeeze(0)
    
    # Convert from [C, H, W] to [H, W, C] for PIL
    if tensor_image.shape[0] == 3:  # If channels-first format
        tensor_image = tensor_image.permute(1, 2, 0)
    
    # Convert to numpy and ensure proper range [0, 255]
    np_image = tensor_image.numpy()
    np_image = np.clip(np_image * 255, 0, 255).astype(np.uint8)
    
    # Create PIL image
    pil_image = Image.fromarray(np_image)
    return pil_image


def convert_pil_to_tensor(pil_image, add_batch_dim=False, device='cpu'):
    """
    Convert a PIL Image to a PyTorch tensor.
    
    Args:
        pil_image: PIL Image object
        add_batch_dim: Whether to add a batch dimension
        device: Device to put the tensor on
        
    Returns:
        PyTorch tensor with shape [C, H, W] or [1, C, H, W] if add_batch_dim=True
    """
    # Convert PIL to numpy array [H, W, C]
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Convert to tensor and rearrange dimensions to [C, H, W]
    tensor_image = torch.from_numpy(np_image).permute(2, 0, 1)
    
    # Add batch dimension if requested
    if add_batch_dim:
        tensor_image = tensor_image.unsqueeze(0)
    
    # Move to specified device
    tensor_image = tensor_image.to(device)
    
    return tensor_image


def save_attack_results(tensor_image, base_path, model_id):
    """
    Save tensor version of an attack result in a model-specific folder.
    
    Args:
        tensor_image: PyTorch tensor containing image data
        base_path: Base path without extension for saving files
        model_id: Model identifier used for folder organization
    
    Returns:
        Path to the saved tensor file
    """
    # Ensure the tensor is detached from computation graph and moved to CPU
    if tensor_image.requires_grad:
        tensor_image = tensor_image.detach()
    
    if tensor_image.device.type != 'cpu':
        tensor_image = tensor_image.cpu()
    
    # Get the directory and base filename
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    
    # Create model-specific subfolder in the attack directory
    model_name = model_id.split('/')[-1].lower() if '/' in model_id else model_id.lower()
    model_directory = os.path.join(directory, model_name)
    os.makedirs(model_directory, exist_ok=True)
    
    # Create full path for tensor file only
    tensor_path = os.path.join(model_directory, f"{filename}.pt")
    
    # Save tensor version
    torch.save(tensor_image, tensor_path)
    print(f"Saved tensor image to {tensor_path}")
    
    return tensor_path

# Helper function to load and process images
def load_image(image_input):
    """
    Load an image from various input types.
    
    Args:
        image_input: Can be a string path to an image file, a PIL Image, 
                    or a PyTorch tensor with shape [C, H, W] or [1, C, H, W]
    
    Returns:
        PIL Image object
    """
    if isinstance(image_input, str):
        # Path to image file
        return Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        # Already a PIL Image
        return image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    elif isinstance(image_input, torch.Tensor):
        # PyTorch tensor
        if image_input.requires_grad:
            image_input = image_input.detach()
        
        if image_input.device.type != 'cpu':
            image_input = image_input.cpu()
        
        # Remove batch dimension if present
        if image_input.dim() == 4:
            image_input = image_input.squeeze(0)
        
        # Convert from [C, H, W] to [H, W, C] for PIL
        if image_input.shape[0] == 3:  # If channels-first format
            image_input = image_input.permute(1, 2, 0)
        
        # Convert to numpy and ensure proper range [0, 255]
        np_image = image_input.numpy()
        np_image = np.clip(np_image * 255, 0, 255).round().astype(np.uint8)
        
        # Create PIL image
        return Image.fromarray(np_image)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

def process_image_input(image_input, processor=None, model_device=None):
    """
    Process image input based on its type.
    
    Args:
        image_input: String path, PIL Image, or PyTorch tensor
        processor: Model processor (if needed for tensor normalization)
        model_device: Device to place tensor on
        
    Returns:
        Tuple of (processed_input, is_raw_tensor, tensor_image)
        - processed_input: PIL Image for proper token generation
        - is_raw_tensor: Boolean indicating if original input was a tensor
        - tensor_image: The original tensor if input was tensor, None otherwise
    """
    is_tensor = isinstance(image_input, torch.Tensor)
    tensor_image = None
    
    if is_tensor:
        # For tensor inputs, create a PIL image for token generation
        # but also keep the original tensor for later substitution
        if image_input.requires_grad:
            tensor_image = image_input.detach()
        else:
            tensor_image = image_input.clone()
            
        # Move to right device if needed
        if model_device and tensor_image.device.type != model_device:
            tensor_image = tensor_image.to(model_device)
            
        # Create a PIL version for token generation
        pil_image = convert_tensor_to_pil(image_input)
        return pil_image, True, tensor_image
    else:
        # For path or PIL image, load normally
        return load_image(image_input), False, None

def resize_images_with_padding(directory_path, target_size=(224, 224), fill_color=(255, 255, 255)):
    """
    Resize all images in a directory to the specified size, preserving aspect ratio
    and adding padding to reach the target size. Overwrites the original files.
    
    Args:
        directory_path: Path to directory containing images
        target_size: Tuple (width, height) for target size
        fill_color: Tuple (R, G, B) for padding color (white by default)
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # Get all image files in directory
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) and 
                  os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(directory_path, img_file)
        try:
            # Open image
            img = Image.open(img_path).convert("RGB")
            
            # Get current dimensions
            width, height = img.size
            
            # Calculate aspect ratios
            target_ratio = target_size[0] / target_size[1]
            img_ratio = width / height
            
            if img_ratio > target_ratio:
                # Image is wider than target ratio, fit to width
                new_width = target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller than target ratio, fit to height
                new_height = target_size[1]
                new_width = int(new_height * img_ratio)
            
            # Resize image preserving aspect ratio
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create new white image with target size
            new_img = Image.new('RGB', target_size, fill_color)
            
            # Calculate position to paste (center)
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            
            # Paste resized image onto padded background
            new_img.paste(img_resized, (paste_x, paste_y))
            
            # Save back to original path (overwriting)
            new_img.save(img_path)
            print(f"Resized {img_file} to {target_size}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Finished processing {len(image_files)} images in {directory_path}")

def update_image_data_json(attack_file, model_id, model_output, attack_params=None, json_path=None):
    """
    Update the image_data.json file with model inference results for an attack image.
    Uses the new hierarchical data structure.
    
    Args:
        attack_file: Name of the attack pt file (without extension)
        model_id: The model identifier used for inference
        model_output: The text output from the model
        attack_params: Optional dictionary of attack parameters
        json_path: Path to the json file (if None, uses default path)
    
    Returns:
        Dict containing the updated JSON data
    """
    # Use current script directory as reference to find images directory
    script_path = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_path, "../images")
    
    # Set default json_path if not provided
    if json_path is None:
        json_path = os.path.join(image_dir, "image_data.json")
    
    # Load existing JSON data or create empty structure
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                data = json.load(file)
        else:
            data = {"images": {}}
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {json_path}, creating new structure")
        data = {"images": {}}
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        data = {"images": {}}
    
    # Ensure images dict exists
    if "images" not in data:
        data["images"] = {}
    
    # Extract model name for easier identification
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    
    # Extract the clean image name (remove model folder path if present)
    if '/' in attack_file:
        # We have a path like 'model_name/image_name', extract just image_name
        base_name = os.path.basename(attack_file)
    else:
        base_name = attack_file
    
    # Get the image name (this is what we use as key in the images dict)
    image_name = base_name
    
    # Ensure the image entry exists
    if image_name not in data["images"]:
        # If this is a new image, create all required structures
        data["images"][image_name] = {
            "metadata": {
                "filename": f"{image_name}.png",
                "question": "",  # Default empty question
                "correct_answer": "",  # Default empty answer
                "target_answer": ""  # Default empty target
            },
            "clean": {
                "path": f"clean/{image_name}.png",
                "model_outputs": {}
            },
            "attacks": {}
        }
    
    # Create timestamp for tracking when inference was run
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create or update the attack entry for this model
    if model_name not in data["images"][image_name]["attacks"]:
        # Initialize attack entry
        data["images"][image_name]["attacks"][model_name] = {
            "metadata": {
                "model_used": model_id,
                "path": f"attack/{model_name}/{image_name}.pt",
                "attack_parameters": attack_params or {},
                "generation_date": timestamp
            },
            "model_outputs": {}
        }
    
    # Add or update the model output for this attack
    data["images"][image_name]["attacks"][model_name]["model_outputs"][model_id] = {
        "output": model_output,
        "timestamp": timestamp
    }
    
    # Write updated data back to file
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Updated {json_path} with attack results from {model_name}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
    
    return data

def update_clean_image_output(image_name, model_id, model_output, json_path=None):
    """
    Update the image_data.json file with model inference results for a clean image.
    Uses the new hierarchical data structure.
    
    Args:
        image_name: Name of the clean image file
        model_id: The model identifier used for inference
        model_output: The text output from the model
        json_path: Path to the json file (if None, uses default path)
    
    Returns:
        Dict containing the updated JSON data
    """
    # Use current script directory as reference to find images directory
    script_path = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_path, "../images")
    
    # Set default json_path if not provided
    if json_path is None:
        json_path = os.path.join(image_dir, "image_data.json")
    
    # Load existing JSON data or create empty structure
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                data = json.load(file)
        else:
            data = {"images": {}}
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {json_path}, creating new structure")
        data = {"images": {}}
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        data = {"images": {}}
    
    # Ensure images dict exists
    if "images" not in data:
        data["images"] = {}
    
    # Extract just the filename without path or extension
    base_name = get_clean_filename(image_name) if '.' in image_name else image_name
    
    # Ensure the image entry exists
    if base_name not in data["images"]:
        # If this is a new image, create all required structures
        data["images"][base_name] = {
            "metadata": {
                "filename": f"{base_name}.png",
                "question": "",  # Default empty question
                "correct_answer": "",  # Default empty answer
                "target_answer": ""  # Default empty target
            },
            "clean": {
                "path": f"clean/{base_name}.png",
                "model_outputs": {}
            },
            "attacks": {}
        }
    
    # Create timestamp for tracking when inference was run
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add or update the model output for the clean image
    data["images"][base_name]["clean"]["model_outputs"][model_id] = {
        "output": model_output,
        "timestamp": timestamp
    }
    
    # Write updated data back to file
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f"Updated clean image outputs for {base_name} with {model_id}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
    
    return data

def get_attack_parameters(input_dict):
    """
    Extract attack parameters from an input dictionary, using standard parameter names.
    
    Args:
        input_dict: Dictionary containing attack parameters
        
    Returns:
        Formatted attack parameter dictionary with standardized keys
    """
    # Map common parameter names to standardized keys
    params = {}
    
    # Extract common parameters with fallbacks
    params["epsilon"] = input_dict.get("epsilon", 0.1)
    params["alpha"] = input_dict.get("alpha_max", input_dict.get("alpha", 0.01))
    params["iterations"] = input_dict.get("num_iter", input_dict.get("iterations", 200))
    
    # Include other parameters if they exist
    if "scheduler_type" in input_dict:
        params["scheduler"] = input_dict["scheduler_type"]
    if "warmup_ratio" in input_dict:
        params["warmup_ratio"] = input_dict["warmup_ratio"]
    
    return params

def get_clean_filename(image_path):
    """
    Extract the filename without extension from a path.
    
    Args:
        image_path: Path to an image file
    
    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(image_path))[0]