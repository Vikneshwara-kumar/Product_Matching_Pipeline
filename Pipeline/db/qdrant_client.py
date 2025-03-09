"""
qdrant_client.py
----------------
Client wrapper to interact with Qdrant vector database, supporting multiple collections.
This version uses the official QdrantClient to perform a search.
"""

import asyncio
import requests
import numpy as np
from utils.logger import log_event_sync
from qdrant_client import QdrantClient

# Create a global QdrantClient instance.
CLIENT = QdrantClient(host="localhost", port=6333)

async def search_embedding(embedding: np.ndarray, collection: str, top_k: int = 5):
    """
    Searches Qdrant for the closest matching embedding in the specified collection.
    
    Args:
        embedding (np.ndarray): The embedding vector to search for.
        collection (str): The name of the Qdrant collection (e.g., "product_image" or "product_text").
        top_k (int): The number of top results to return.
        
    Returns:
        product_id: Identifier of the top matched product from the specified collection.
        
    Raises:
        ValueError: If no matching product is found.
    """
    try:
        query_vector = embedding.tolist()[0]

        # Since QdrantClient is synchronous, wrap the search call in asyncio.to_thread.
        def do_search():
            return CLIENT.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k
            )
        hits = await asyncio.to_thread(do_search)

        if hits and len(hits) > 0:
            # Assuming each hit is an object with an 'id' attribute.
            product_id = hits[0].payload["id"]
            match_score =hits[0].score
            log_event_sync(
                "INFO",
                f"Successfully retrieved product from collection '{collection}'.",
                extra={"collection": collection, "product_id": product_id}
            )
            return match_score, product_id
        else:
            msg = f"No matching product found in collection '{collection}'."
            log_event_sync(
                "ERROR",
                msg,
                extra={"collection": collection, "query_vector": query_vector}
            )
            raise ValueError(msg)
            
    except Exception as e:
        log_event_sync(
            "ERROR",
            f"Error during Qdrant search in collection '{collection}': {e}",
            extra={"collection": collection, "query_vector": query_vector}
        )
        raise e
