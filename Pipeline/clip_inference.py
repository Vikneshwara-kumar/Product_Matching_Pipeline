"""
clip_inference.py
-----------------
Triton client wrappers for obtaining CLIP embeddings separately:
  - get_clip_text_embedding: Uses the "clip_text" model.
  - get_clip_visual_embedding: Uses the "clip_visual" model.
This version uses PIL for images and Hugging Face's CLIPProcessor for preprocessing.
"""

import asyncio
import numpy as np
import tritonclient.http as httpclient  # NVIDIA Triton client library
from utils.logger import log_event_sync  # our MongoDB logger
from transformers import CLIPProcessor, CLIPTokenizer

# Global processor and tokenizer instances.

PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

trition_url = "localhost:8000"  # Triton Inference Server endpoint
TCLIENT = httpclient.InferenceServerClient(url=trition_url)

async def get_clip_text_embedding(text_prompt: str) -> np.ndarray:
    """
    Given a text prompt, obtain the text embedding from the clip_text model.
    
    Args:
        text_prompt (str): The text input.
    
    Returns:
        np.ndarray: The text embedding.
    """
    try:
        # Preprocess text using Hugging Face's tokenizer.
        text_data = preprocess_text(text_prompt)
    except Exception as e:
        log_event_sync("ERROR", f"Text preprocessing failed: {e}", extra={"function": "get_clip_text_embedding", "text": text_prompt})
        raise ValueError(f"Text preprocessing failed: {e}")

    try:
        # Create Triton client.
        client = TCLIENT
    except Exception as e:
        log_event_sync("ERROR", f"Failed to create Triton client for text: {e}", extra={"function": "get_clip_text_embedding"})
        raise ConnectionError(f"Failed to create Triton client: {e}")

    try:
        # Prepare input for the clip_text model.
        text_inputs = [
            httpclient.InferInput("input", text_data.shape, "INT64")
        ]
        text_inputs[0].set_data_from_numpy(text_data)
    
        text_outputs = [
            httpclient.InferRequestedOutput("output")
        ]
    
        # Perform inference asynchronously.
        text_result = await asyncio.to_thread(
            client.infer, model_name="clip_text", model_version="1", inputs=text_inputs, outputs=text_outputs
        )
    
        text_embedding = text_result.as_numpy("output")
        if text_embedding is None:
            raise ValueError("Triton returned an empty result for text inference.")
    except Exception as e:
        log_event_sync("ERROR", f"Error during text inference: {e}", extra={"function": "get_clip_text_embedding"})
        raise RuntimeError(f"Error during text inference: {e}")

    return text_embedding

async def get_clip_visual_embedding(image) -> np.ndarray:
    """
    Given a PIL image, obtain the visual embedding from the clip_visual model.
    
    Args:
        image (PIL.Image.Image): The input image.
    
    Returns:
        np.ndarray: The visual embedding.
    """
    try:
        # Preprocess image using Hugging Face's CLIPProcessor.
        image_data = preprocess_image(image)
    except Exception as e:
        log_event_sync("ERROR", f"Image preprocessing failed: {e}", extra={"function": "get_clip_visual_embedding"})
        raise RuntimeError(f"Image preprocessing failed: {e}")

    try:
        # Create Triton client.
        client = TCLIENT
    except Exception as e:
        log_event_sync("ERROR", f"Failed to create Triton client for image: {e}", extra={"function": "get_clip_visual_embedding"})
        raise ConnectionError(f"Failed to create Triton client: {e}")

    try:
        # Prepare input for the clip_visual model.
        visual_inputs = [
            httpclient.InferInput("Input_Image", image_data.shape, "FP16")
        ]
        visual_inputs[0].set_data_from_numpy(image_data)
        
        visual_outputs = [
            httpclient.InferRequestedOutput("Image_Embeddings")
        ]
        
        # Perform inference asynchronously.
        visual_result = await asyncio.to_thread(
            client.infer, model_name="clip_visual", model_version="1", inputs=visual_inputs, outputs=visual_outputs
        )
        
        visual_embedding = visual_result.as_numpy("Image_Embeddings")
    except Exception as e:
        log_event_sync("ERROR", f"Error during image inference: {e}", extra={"function": "get_clip_visual_embedding"})
        raise RuntimeError(f"Error during image inference: {e}")

    return visual_embedding

def preprocess_image(image):
    """
    Preprocess the PIL image into the required numpy array format using Hugging Face's CLIPProcessor.
    
    Returns:
        A numpy array with shape [1, 3, 224, 224] and dtype float16.
    """
    try:
        # Check if the input is a valid PIL Image.
        from PIL import Image
        if not isinstance(image, Image.Image):
            raise ValueError("Input is not a valid PIL Image.")

        # Use the processor to preprocess the image.
        inputs = PROCESSOR(images=image, return_tensors="np")
        if "pixel_values" not in inputs:
            raise ValueError("Processor output missing key 'pixel_values'.")

        image_array = inputs["pixel_values"]

        # Convert to numpy array if not already, then cast to float16.
        if isinstance(image_array, np.ndarray):
            image_np = image_array.astype(np.float16)
        elif hasattr(image_array, "numpy"):
            image_np = image_array.numpy().astype(np.float16)
        else:
            raise ValueError("The output 'pixel_values' is neither a numpy array nor convertible to one.")
    
        # Add a batch dimension: [1, 3, 224, 224]
        if image_np.ndim == 3:
            # No batch dimension, so add one.
            image_np = np.expand_dims(image_np, axis=0)
        elif image_np.ndim == 5 and image_np.shape[1] == 1:
            # Remove the extra dimension (squeeze out axis 1)
            image_np = np.squeeze(image_np, axis=1)
        # Otherwise, if image_np.ndim is 4, assume it's already [batch, 3, 224, 224].   

        return image_np

    except Exception as e:
        log_event_sync("ERROR", f"Error in preprocessing Image: {e}", extra={"function": "preprocess_image" })
        raise RuntimeError(f"Error in preprocessing image: {e}")

def preprocess_text(text):
    """
    Preprocess the text prompt into the required numpy array format using Hugging Face's CLIPTokenizer.
    
    Returns:
        A numpy array representing the tokenized text (e.g. shape [1, 77] or similar).
    """
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Use the tokenizer to tokenize the text.
        inputs = TOKENIZER(text, return_tensors="np", padding="max_length", max_length=77, truncation=True)
        if "input_ids" not in inputs:
            raise ValueError("Tokenizer output missing key 'input_ids'.")
        text_array = inputs["input_ids"].astype(np.int64)

        return text_array

    except Exception as e:
        log_event_sync("ERROR", f"Error in preprocessing text: {e}", extra={"function": "preprocess_text", "text": text})
        raise RuntimeError(f"Error in preprocessing text: {e}")
