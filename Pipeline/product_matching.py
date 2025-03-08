"""
product_matching.py
-------------------
Handles product matching logic by combining Qdrant and MongoDB queries.
Includes an in-memory cache to optimize repeated queries.
"""

import asyncio
import numpy as np
import hashlib
from db import qdrant_client, mongodb_client
from utils.logger import log_event_sync  # our MongoDB logger

# In-memory cache for matching results.
# Keys are hashes of embeddings; values are product metadata.
product_cache = {}
cache_lock = asyncio.Lock()

def hash_embedding(embedding: np.ndarray) -> str:
    """
    Computes an MD5 hash for a given embedding.
    This hash is used as the cache key.
    """
    m = hashlib.md5()
    m.update(embedding.tobytes())
    return m.hexdigest()

async def match_product_by_text(text_embedding: np.ndarray):
    """
    Matches a product using a text embedding by querying the Qdrant 'product_text' collection,
    then retrieving metadata from MongoDB.
    Uses an in-memory cache to avoid redundant queries.

    Args:
        text_embedding (np.ndarray): The text embedding vector.
    
    Returns:
        dict: The product metadata.
    """
    cache_key = "text_" + hash_embedding(text_embedding)
    async with cache_lock:
        if cache_key in product_cache:
            log_event_sync("INFO", f"Cache hit for text embedding.", extra={"cache_key": cache_key})
            return product_cache[cache_key]

    try:
        product_id = await qdrant_client.search_embedding(text_embedding, collection="product_text")
    except Exception as e:
        log_event_sync("ERROR", f"Error during text matching: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Text matching failed: {e}")

    try:
        product = await mongodb_client.get_product(product_id)
    except Exception as e:
        log_event_sync("ERROR", f"Error retrieving product metadata for product id {product_id}: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Error retrieving product metadata for product id {product_id}: {e}")

    async with cache_lock:
        product_cache[cache_key] = product

    return product

async def match_product_by_visual(visual_embedding: np.ndarray):
    """
    Matches a product using a visual embedding by querying the Qdrant 'product_image' collection,
    then retrieving metadata from MongoDB.
    Uses an in-memory cache to optimize repeated queries.

    Args:
        visual_embedding (np.ndarray): The visual embedding vector.
    
    Returns:
        dict: The product metadata.
    """
    cache_key = "visual_" + hash_embedding(visual_embedding)
    async with cache_lock:
        if cache_key in product_cache:
            log_event_sync("INFO", f"Cache hit for visual embedding.", extra={"cache_key": cache_key})
            return product_cache[cache_key]

    try:
        product_id = await qdrant_client.search_embedding(visual_embedding, collection="product_image")
    except Exception as e:
        log_event_sync("ERROR", f"Error during visual matching: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Visual matching failed: {e}")

    try:
        product = await mongodb_client.get_product(product_id)
    except Exception as e:
        log_event_sync("ERROR", f"Error retrieving product metadata for product id {product_id}: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Error retrieving product metadata for product id {product_id}: {e}")

    async with cache_lock:
        product_cache[cache_key] = product

    return product
