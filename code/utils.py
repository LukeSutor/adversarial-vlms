import torch
import numpy as np
from PIL import Image
import os

def save_image_tensor(tensor_image, output_path="perturbed_tensor.pt"):
    """
    Save the image as a PyTorch tensor file (.pt) to preserve exact pixel values
    without quantization errors.
    
    Args:
        tensor_image: PyTorch tensor containing image data
        output_path: Path where to save the tensor file
    """
    # Ensure the tensor is detached from computation graph and moved to CPU
    if tensor_image.requires_grad:
        tensor_image = tensor_image.detach()
    
    if tensor_image.device.type != 'cpu':
        tensor_image = tensor_image.cpu()
    
    # Save the tensor
    torch.save(tensor_image, output_path)
    print(f"Saved tensor image to {output_path}")


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


def save_image(perturbed_image, output_path="perturbed_image.png", save_tensor=False):
    """
    Save perturbed image with special handling for robustness.
    Can save both as PNG and as PyTorch tensor.
    
    Args:
        perturbed_image: NumPy array or PyTorch tensor containing the image
        output_path: Path for the output file
        save_tensor: Whether to also save the tensor version
    """
    # Check if input is a tensor
    is_tensor = isinstance(perturbed_image, torch.Tensor)
    
    # Handle tensor input
    if is_tensor:
        # Save tensor version if requested (before any transformations)
        if save_tensor:
            tensor_path = os.path.splitext(output_path)[0] + '.pt'
            save_image_tensor(perturbed_image, tensor_path)
        
        # Convert to numpy for PNG saving
        if perturbed_image.requires_grad:
            perturbed_image = perturbed_image.detach()
        
        if perturbed_image.dim() == 4:  # If has batch dimension [1, C, H, W]
            perturbed_image = perturbed_image.squeeze(0)
            
        # Rearrange from [C, H, W] to [H, W, C]
        perturbed_image = perturbed_image.permute(1, 2, 0).cpu().numpy()
    else:
        # For numpy input, still save tensor version if requested
        if save_tensor:
            tensor_path = os.path.splitext(output_path)[0] + '.pt'
            tensor_image = torch.from_numpy(perturbed_image.transpose(2, 0, 1).copy())
            save_image_tensor(tensor_image, tensor_path)
    
    # Convert the image to 8-bit format
    perturbed_image_8bit = np.clip(perturbed_image * 255, 0, 255).round().astype(np.uint8)
    
    # Create PIL image
    image = Image.fromarray(perturbed_image_8bit)
    
    # Save with no compression to preserve details
    image.save(output_path, format='PNG', compress_level=0, optimize=False)
    print(f"Saved attack image to {output_path}")

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