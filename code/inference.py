from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, PaliGemmaProcessor, Qwen2_5_VLForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import numpy as np
from typing import Dict, Optional, Any, Union
import shutil
import pathlib
from utils import load_image, convert_tensor_to_pil

# Default model cache directory
DEFAULT_CACHE_DIR = "/blue/rcstudents/luke.sutor/adversarial-vlms/models/"

class ModelManager:
    """Manages loading and caching of models to optimize VRAM usage."""
    
    def __init__(self, device="cuda:0", dtype=torch.float16, cache_dir=DEFAULT_CACHE_DIR):
        self.device = device
        self.dtype = dtype
        self.current_model = None
        self.current_model_id = None
        self.current_tokenizer = None
        self.current_processor = None
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        self._ensure_cache_directory()
    
    def set_cache_directory(self, new_cache_dir):
        """Update the cache directory for model storage"""
        self.cache_dir = new_cache_dir
        self._ensure_cache_directory()
        return self.cache_dir
    
    def _ensure_cache_directory(self):
        """Make sure the cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Created model cache directory: {self.cache_dir}")
    
    def get_model(self, model_id: str):
        """Get a model, loading it if necessary and unloading the previous model."""
        if self.current_model_id == model_id:
            return self.current_model, self.current_tokenizer, self.current_processor
        
        # Unload current model if any
        if self.current_model is not None:
            self._unload_current_model()
        
        # Load requested model
        model, tokenizer, processor = self._load_model(model_id)
        self.current_model = model
        self.current_model_id = model_id
        self.current_tokenizer = tokenizer
        self.current_processor = processor
        
        return model, tokenizer, processor
    
    def _unload_current_model(self):
        """Unload the current model to free up VRAM."""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            del self.current_processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.current_model = None
            self.current_model_id = None
            self.current_tokenizer = None
            self.current_processor = None
    
    def _load_model(self, model_id: str):
        """Load a specific model, tokenizer, and processor."""
        print(f"Loading model from cache directory: {self.cache_dir}")
        model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        if "paligemma" in model_id.lower():
            # Note: Using float16 instead of bfloat16 for all PaliGemma models
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                cache_dir=self.cache_dir
            ).eval()
            
            # Use specific PaliGemmaProcessor for PaliGemma models
            if "paligemma2" in model_id.lower():
                processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
                # For PaliGemma2 models, we still need a tokenizer for some operations
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
                processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        elif "qwen2.5-vl" in model_id.lower():
            # Special handling for Qwen2.5-VL models
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                cache_dir=self.cache_dir
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        elif "qwen2-vl" in model_id.lower():
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                cache_dir=self.cache_dir
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
        elif "llama-3.2" in model_id.lower() and "vision" in model_id.lower():
            # Special handling for Llama 3.2 Vision models
            model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                cache_dir=self.cache_dir
            ).eval()
            
            # Llama models use AutoProcessor and don't need a separate tokenizer
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
            # Reuse processor as tokenizer - it contains the tokenizer functionality
            tokenizer = processor
        elif "llava" in model_id.lower():
            # Special handling for LLaVA models
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
                cache_dir=self.cache_dir
            ).eval()
            
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.cache_dir)
            # Get the tokenizer from the processor for LLaVA models
            tokenizer = processor.tokenizer
            
            return model, tokenizer, processor
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        
        return model, tokenizer, processor
    
    def get_model_path(self, model_id):
        """Get the local path for a model"""
        # This is a simplified approach - real path depends on transformers' internal naming
        return os.path.join(self.cache_dir, model_id.replace('/', '--'))
    
    def clear_cache(self, model_id=None):
        """
        Remove cached model files to free disk space
        If model_id is None, clears the entire cache directory
        """
        if model_id is None:
            # Clear all cached models (dangerous, confirm first)
            confirm = input(f"Are you sure you want to delete ALL models in {self.cache_dir}? (yes/no): ")
            if confirm.lower() == 'yes':
                for item in os.listdir(self.cache_dir):
                    item_path = os.path.join(self.cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                print(f"Cleared all cached models from {self.cache_dir}")
        else:
            # Clear specific model
            model_path = self.get_model_path(model_id)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"Removed cached model: {model_id}")
            else:
                print(f"Model not found in cache: {model_id}")


# Initialize the model manager as a singleton with the default cache directory
model_manager = ModelManager(cache_dir=DEFAULT_CACHE_DIR)

# Function to change the cache directory
def set_models_directory(new_directory):
    """Set the directory where models will be cached"""
    return model_manager.set_cache_directory(new_directory)

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

def run_inference_paligemma(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 100
) -> str:
    """Run inference with a PaliGemma model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    # Process input differently based on type
    image, is_tensor, tensor_image = process_image_input(image_input, processor, model.device)
    
    # Always use processor with PIL image to get the correct format with image tokens
    model_input = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    
    # If original input was tensor, substitute with the exact tensor values
    if is_tensor:
        model_input['pixel_values'] = tensor_image
    
    # Remember the input length for decoding
    input_len = model_input["input_ids"].shape[-1]

    # Generate output
    with torch.inference_mode():
        generation = model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)
        final_generation = generation[0][input_len:]
        result = processor.decode(final_generation, skip_special_tokens=True)
    
    return result


def run_inference_qwen2_vl(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 128
) -> str:
    """Run inference with a Qwen2-VL model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    # Process input differently based on type
    image, is_tensor, tensor_image = process_image_input(image_input, processor, model.device)
    
    # Always use the processor with PIL image to get proper image tokens
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
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
    
    # If original input was tensor, substitute the exact tensor values
    if is_tensor:
        inputs['pixel_values'] = tensor_image
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


def run_inference_qwen25_vl(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 128
) -> str:
    """Run inference with a Qwen2.5-VL model."""
    # Use same implementation as Qwen2-VL as they have the same interface
    return run_inference_qwen2_vl(model_id, prompt, image_input, max_new_tokens)


def run_inference_llama(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 100
) -> str:
    """Run inference with a Llama-3.2-Vision model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    # Process input differently based on type
    image, is_tensor, tensor_image = process_image_input(image_input, processor, model.device)
    
    # Always process with PIL image to get proper structure
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    # Process inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    )
    
    # If original input was tensor, substitute the exact tensor values
    if is_tensor:
        inputs['pixel_values'] = tensor_image
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode the whole sequence as Llama doesn't need to trim input tokens
        result = processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Remove the input prompt from the result
        if prompt in result:
            # Find the last instance of the prompt (after any system messages)
            prompt_pos = result.rfind(prompt)
            if prompt_pos > -1:
                result = result[prompt_pos + len(prompt):].strip()
    
    return result

def run_inference_llava(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 100
) -> str:
    """Run inference with a LLaVA model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    # Process input differently based on type
    image, is_tensor, tensor_image = process_image_input(image_input, processor, model.device)
    
    # Always process with PIL image to get proper structure with image tokens
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Apply chat template with direct tokenization
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    
    # Process image to get proper structure
    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values
    inputs["pixel_values"] = pixel_values
    
    # If original input was tensor, substitute the exact tensor values
    if is_tensor:
        inputs["pixel_values"] = tensor_image
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        result = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Extract just the assistant's response
        if prompt in result:
            # Find the last instance of the prompt
            prompt_pos = result.rfind(prompt)
            if prompt_pos > -1:
                result = result[prompt_pos + len(prompt):].strip()
    
    return result

def get_inference_function(model_id: str):
    """Return the appropriate inference function for the given model ID."""
    if "paligemma" in model_id.lower():
        return run_inference_paligemma
    elif "qwen2.5-vl" in model_id.lower():
        return run_inference_qwen25_vl
    elif "qwen2-vl" in model_id.lower():
        return run_inference_qwen2_vl
    elif "llama-3.2" in model_id.lower() and "vision" in model_id.lower():
        return run_inference_llama
    elif "llava" in model_id.lower():
        return run_inference_llava
    else:
        raise ValueError(f"No inference function available for model: {model_id}")


def run_inference(
    model_id: str,
    prompt: str,
    image_input: Union[str, Image.Image, torch.Tensor],
    max_new_tokens: int = 100
) -> str:
    """
    Run inference on a specified model with the given prompt and image.
    
    Args:
        model_id: Hugging Face model ID (e.g., "google/paligemma-3b-mix-224")
        prompt: Text prompt to use with the image
        image_input: Can be a string path to an image, a PIL Image, or a PyTorch tensor
                    When using a tensor, it will be passed directly to avoid quantization
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text response
    """
    inference_fn = get_inference_function(model_id)
    return inference_fn(model_id, prompt, image_input, max_new_tokens=max_new_tokens)


# Predefined model constants for convenience
class Models:
    PALIGEMMA_3B = "google/paligemma-3b-mix-224"
    PALIGEMMA2_3B = "google/paligemma2-3b-mix-224"
    PALIGEMMA2_10B = "google/paligemma2-10b-mix-224"
    PALIGEMMA2_28B = "google/paligemma2-28b-mix-224"
    QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
    QWEN25_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN25_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN25_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"
    LLAMA_3_2_11B_VISION = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    LLAVA_1_5_7B = "llava-hf/llava-1.5-7b-hf"
    LLAVA_1_5_13B = "llava-hf/llava-1.5-13b-hf"
    # More models can be added here as constants


if __name__ == "__main__":
    # Define the model ID and input parameters
    models = [Models.PALIGEMMA2_10B]
    prompt = "Answer the question"
    max_new_tokens = 100

    # Get the local path to the images
    script_path = "/".join(__file__.split("/")[:-1])
    image_dir = os.path.join(script_path, "../images")
    attack_dir = os.path.join(image_dir, "attack")
    clean_dir = os.path.join(image_dir, "clean")

    image_path = os.path.join(attack_dir, "math.png")
    
    # Example with image path
    print("\n=== Running inference with image path ===")
    
    # Run inference with image path
    for model in models:
        try:
            result = run_inference(model, prompt, image_path, max_new_tokens)
            print("Generated Response:")
            print(result)
        except Exception as e:
            print(f"An error occurred during inference: {e}")
    
    # Example with tensor
    tensor_path = os.path.splitext(image_path)[0] + '.pt'
    if os.path.exists(tensor_path):
        print("\n=== Running inference with tensor directly (no quantization) ===")
        try:
            # Load tensor
            tensor_image = torch.load(tensor_path)
            
            # Run inference with tensor directly (bypassing PIL conversion)
            for model in models:
                result = run_inference(model, prompt, tensor_image, max_new_tokens)
                print("Generated Response (from raw tensor, preserving exact values):")
                print(result)
        except Exception as e:
            print(f"An error occurred during tensor inference: {e}")