"""
main.py
--------
Main application for product matching.
"""

import asyncio
import sys
from PIL import Image
import io

from clip_inference import get_clip_embeddings
from product_matching import match_product
from utils.logger import log_event

async def process_request(image_bytes: bytes):
    try:
        # Load input image (PIL can also process in-memory bytes)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Obtain embeddings from the Triton Inference Server (both textual & visual)
        text_embedding, visual_embedding = await get_clip_embeddings(image=image, text_prompt="A product image")
        
        # Find the matching product using Qdrant and retrieve metadata from MongoDB
        product = await match_product(text_embedding, visual_embedding)
        
        # Log success event
        log_event("INFO", "Product match successful", extra={"product_id": product.get("id")})
        return product

    except Exception as e:
        # Log error to the logging MongoDB
        log_event("ERROR", "Failed to process request", extra={"error": str(e)})
        return {"error": str(e)}

if __name__ == "__main__":
    # For demonstration purposes, read an image file passed as argument.
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_file_path>")
        sys.exit(1)
    
    image_file_path = sys.argv[1]
    with open(image_file_path, "rb") as f:
        image_bytes = f.read()

    product_result = asyncio.run(process_request(image_bytes))
    print("Matched Product:")
    print(product_result)
