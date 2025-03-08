"""
product_matching.py
-------------------
Handles the product matching logic by combining Qdrant and MongoDB queries.
"""

import asyncio
from db import qdrant_client, mongodb_client

async def match_product(text_embedding, visual_embedding):
    """
    Matches an input product by performing a nearest neighbor search on Qdrant
    using combined text and visual embeddings. Retrieves metadata from MongoDB.
    """
    # Option 1: Concatenate embeddings, or perform separate queries
    combined_embedding = combine_embeddings(text_embedding, visual_embedding)
    
    # Query Qdrant for nearest neighbor (assume top-1 match)
    product_id = await qdrant_client.search_embedding(combined_embedding)
    
    # Retrieve product metadata from MongoDB
    product = await mongodb_client.get_product(product_id)
    
    return product

def combine_embeddings(text_emb, visual_emb):
    """
    Combines text and visual embeddings into a single embedding vector.
    This example simply concatenates them; alternative strategies can be used.
    """
    import numpy as np
    return np.concatenate([text_emb, visual_emb], axis=-1)
