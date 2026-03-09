"""
product_matching.py
-------------------
Handles product matching logic by combining Qdrant and MongoDB queries.
Includes an in-memory cache to optimize repeated queries.
"""

import asyncio
import numpy as np
import hashlib
from collections import OrderedDict
from db import qdrant_client, mongodb_client
from qdrant_client import QdrantClient
from utils.logger import log_event_sync  # our MongoDB logger

# In-memory cache for matching results.
# Keys are hashes of embeddings; values are product metadata.
MAX_CACHE_ENTRIES = 1000
product_cache = OrderedDict()
cache_lock = asyncio.Lock()

def hash_embedding(embedding: np.ndarray) -> str:
    """
    Computes a SHA-256 hash for a given embedding.
    This hash is used as the cache key.
    """
    # Use SHA-256 to reduce collision risk for cache keys.
    m = hashlib.sha256()
    m.update(embedding.tobytes())
    return m.hexdigest()


def _cache_set(cache_key: str, value):
    """Insert into bounded cache and evict oldest entry when limit is exceeded."""
    product_cache[cache_key] = value
    product_cache.move_to_end(cache_key)
    if len(product_cache) > MAX_CACHE_ENTRIES:
        product_cache.popitem(last=False)

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
            product_cache.move_to_end(cache_key)
            log_event_sync("INFO", f"Cache hit for text embedding.", extra={"cache_key": cache_key})
            return product_cache[cache_key]

    try:
        product_id = await qdrant_client.search_embedding(text_embedding, collection="products_text")
    except Exception as e:
        log_event_sync("ERROR", f"Error during text matching: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Text matching failed: {e}")

    try:
        product = await mongodb_client.get_product(product_id)
    except Exception as e:
        log_event_sync("ERROR", f"Error retrieving product metadata for product id {product_id}: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Error retrieving product metadata for product id {product_id}: {e}")

    async with cache_lock:
        _cache_set(cache_key, product)

    return product

async def match_product_by_visual(visual_embedding: np.ndarray):
    """
    Matches a product using a visual embedding by querying the Qdrant 'products_image' collection,
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
            product_cache.move_to_end(cache_key)
            log_event_sync("INFO", f"Cache hit for visual embedding.", extra={"cache_key": cache_key})
            return product_cache[cache_key]

    try:
        match_score, product_id = await qdrant_client.search_embedding(visual_embedding, collection="products_visual")
    except Exception as e:
        log_event_sync("ERROR", f"Error during visual matching: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Visual matching failed: {e}")

    try:
        product = await mongodb_client.get_product(product_id)
    except Exception as e:
        log_event_sync("ERROR", f"Error retrieving product metadata for product id {product_id}: {e}", extra={"cache_key": cache_key})
        raise RuntimeError(f"Error retrieving product metadata for product id {product_id}: {e}")

    async with cache_lock:
        _cache_set(cache_key, product)

    return match_score, product
