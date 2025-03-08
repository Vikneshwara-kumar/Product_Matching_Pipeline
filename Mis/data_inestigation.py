import os
import json
import torch
import clip
from PIL import Image
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# --------------------------
# 1. Load JSON Metadata
# --------------------------
json_file_path = "metadata.json"  # Update path if needed
with open(json_file_path, "r") as f:
    products = json.load(f)

# --------------------------
# 2. Setup Qdrant Client
# --------------------------
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "products"

# create the collection with vector dimension 512 (for CLIP ViT-B/32)
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=512,
        distance=Distance.COSINE
    )
)

# --------------------------
# 3. Setup MongoDB Client
# --------------------------
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["product_db"]
metadata_collection = db["product_metadata"]

# --------------------------
# 4. Load CLIP Model
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --------------------------
# 5. Process Products & Ingest Data
# --------------------------
for product in products:
    image_path = product.get("image_path")
    
    # Attempt to open and process the image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.encode_image(image_input).cpu().numpy().flatten()
    
    # Insert metadata into MongoDB
    metadata_collection.insert_one(product)
    
    # Prepare the payload for Qdrant. You can customize the payload to include fields
    payload = {
        "id": product["id"],
        "SKU": product["SKU"],
        "name": product["name"],
        "brand": product["brand"],
        "category": product["category"],
        "color": product["color"],
        "price": product["price"]
    }
    
    # Create a point structure for Qdrant
    point = PointStruct(
        id=int(product["id"]) if product["id"].isdigit() else product["id"],
        vector=embedding,
        payload=payload
    )
    
    # Insert point into Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=[point])
    print(f"Processed product {product['id']} - {product['name']}")

print("Data ingestion into Qdrant and MongoDB is complete.")
