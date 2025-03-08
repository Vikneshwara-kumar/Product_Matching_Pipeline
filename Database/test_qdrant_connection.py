# test_qdrant_connection.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

def test_qdrant_connection():
    try:
        # Connect to Qdrant instance (assumes it's running on localhost:6333)
        qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # Define a test collection name
        test_collection = "test_collection"
        
        # Create a dummy collection with a vector dimension of 512 (for CLIP ViT-B/32)
        qdrant_client.recreate_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        print(f"Successfully created collection '{test_collection}' in Qdrant.")
        
        # Optionally, you can list collections if needed:
        # collections = qdrant_client.get_collections()
        # print("Collections:", collections)
    except Exception as e:
        print("Error testing Qdrant connection:", e)

if __name__ == "__main__":
    test_qdrant_connection()
