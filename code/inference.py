from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration, Qwen2VLForConditionalGeneration, PaliGemmaProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
from typing import Dict, Optional, Any, Union

class ModelManager:
    """Manages loading and caching of models to optimize VRAM usage."""
    
    def __init__(self, device="cuda:0", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.current_model = None
        self.current_model_id = None
        self.current_tokenizer = None
        self.current_processor = None
    
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
        if "paligemma" in model_id.lower():
            # Note: Using float16 instead of bfloat16 for all PaliGemma models
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
            ).eval()
            
            # Use specific PaliGemmaProcessor for PaliGemma models
            if "paligemma2" in model_id.lower():
                processor = PaliGemmaProcessor.from_pretrained(model_id)
                # For PaliGemma2 models, we still need a tokenizer for some operations
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                processor = AutoProcessor.from_pretrained(model_id)
        elif "qwen2-vl" in model_id.lower():
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                device_map=self.device,
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        
        return model, tokenizer, processor


# Initialize the model manager as a singleton
model_manager = ModelManager()


def run_inference_paligemma(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int = 100
) -> str:
    """Run inference with a PaliGemma model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    image = Image.open(image_path).convert("RGB")
    model_input = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_input["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_input, max_new_tokens=max_new_tokens, do_sample=False)
        final_generation = generation[0][input_len:]
        result = processor.decode(final_generation, skip_special_tokens=True)
    
    return result


def run_inference_qwen2_vl(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int = 128
) -> str:
    """Run inference with a Qwen2-VL model."""
    model, tokenizer, processor = model_manager.get_model(model_id)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
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
    inputs = inputs.to(model.device)

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""


def get_inference_function(model_id: str):
    """Return the appropriate inference function for the given model ID."""
    if "paligemma" in model_id.lower():
        return run_inference_paligemma
    elif "qwen2-vl" in model_id.lower():
        return run_inference_qwen2_vl
    else:
        raise ValueError(f"No inference function available for model: {model_id}")


def run_inference(
    model_id: str,
    prompt: str,
    image_path: str,
    max_new_tokens: int = 100
) -> str:
    """
    Run inference on a specified model with the given prompt and image.
    
    Args:
        model_id: Hugging Face model ID (e.g., "google/paligemma-3b-mix-224")
        prompt: Text prompt to use with the image
        image_path: Path to the input image
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text response
    """
    inference_fn = get_inference_function(model_id)
    return inference_fn(model_id, prompt, image_path, max_new_tokens=max_new_tokens)


# Predefined model constants for convenience
class Models:
    PALIGEMMA_3B = "google/paligemma-3b-mix-224"
    PALIGEMMA2_3B = "google/paligemma2-3b-mix-224"
    PALIGEMMA2_10B = "google/paligemma2-10b-mix-224"
    PALIGEMMA2_28B = "google/paligemma2-28b-mix-224"
    QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
    # More models can be added here as constants


if __name__ == "__main__":
    # Define the model ID and input parameters
    model_id = Models.PALIGEMMA2_10B
    prompt = "Answer the question"
    image_path = "attack_image.png"
    max_new_tokens = 100

    # Run inference
    try:
        result = run_inference(model_id, prompt, image_path, max_new_tokens)
        print("Generated Response:")
        print(result)
    except Exception as e:
        print(f"An error occurred during inference: {e}")