"""
qdrant_client.py
----------------
Client wrapper to interact with Qdrant vector database, supporting multiple collections.
"""

import asyncio
import requests
import numpy as np
from utils.logger import log_event_sync

QDRANT_URL = "http://localhost:6333"

async def search_embedding(embedding: np.ndarray, collection: str, top_k: int = 1):
    """
    Searches Qdrant for the closest matching embedding in the specified collection.
    
    Args:
        embedding (np.ndarray): The embedding vector to search for.
        collection (str): The name of the Qdrant collection (e.g., "product_image" or "product_text").
        top_k (int): The number of top results to return (default is 1).
        
    Returns:
        product_id: Identifier of the top matched product from the specified collection.
        
    Raises:
        ValueError: If no matching product is found or if the response is invalid.
    """
    payload = {
        "vector": embedding.tolist(),  # Convert numpy array to list.
        "limit": top_k
    }
    url = f"{QDRANT_URL}/collections/{collection}/points/search"
    
    try:
        # Make the asynchronous request to Qdrant.
        response = await asyncio.to_thread(requests.post, url, json=payload)
        results = response.json()
        
        if results.get("result"):
            product_id = results["result"][0]["id"]
            # Log successful retrieval.
            log_event_sync(
                "INFO",
                f"Successfully retrieved product from collection '{collection}'.",
                extra={"collection": collection, "product_id": product_id}
            )
            return product_id
        else:
            msg = f"No matching product found in collection '{collection}'."
            log_event_sync(
                "ERROR",
                msg,
                extra={"collection": collection, "payload": payload}
            )
            raise ValueError(msg)
            
    except Exception as e:
        # Log any exceptions during the search process.
        log_event_sync(
            "ERROR",
            f"Error during Qdrant search in collection '{collection}': {e}",
            extra={"collection": collection, "payload": payload}
        )
        raise e
