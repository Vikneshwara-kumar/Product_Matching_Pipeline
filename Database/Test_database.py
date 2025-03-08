from qdrant_client import QdrantClient
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from pymongo import MongoClient

def delete_mongodb_data():
    try:
        mongo_client = MongoClient("mongodb://localhost:27017/")
        db = mongo_client["product_db"]  # Replace with your database name
        collection_name = "product_metadata"  # Replace with your collection name
        collection = db[collection_name]
        
        # Delete all documents in the collection
        result = collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents from MongoDB collection '{collection_name}'.")
        
        # Alternatively, if you wish to drop the entire collection:
        # collection.drop()
        # print(f"Dropped MongoDB collection '{collection_name}'.")
    except Exception as e:
        print("Error deleting MongoDB data:", e)

def delete_qdrant_data():
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name = "products"  # Change to your collection name

        # Approach 1: Delete all points using a match-all filter
        # Uncomment the following lines if you prefer to clear the collection without deleting it.
        # qdrant_client.delete(
        #     collection_name=collection_name,
        #     points_selector={"filter": Filter(match={})}
        # )
        # print(f"Deleted all points from Qdrant collection '{collection_name}'.")

        # Approach 2: Delete the entire collection
        qdrant_client.delete_collection(collection_name=collection_name)
        print(f"Deleted Qdrant collection '{collection_name}'.")
    except Exception as e:
        print("Error deleting Qdrant data:", e)

def retrieve_qdrant_points():
    try:
        # Connect to Qdrant (assumes it's running on localhost:6333)
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name = "products"  # Use the collection name you created

        # Unpack the returned tuple from scroll: (points, next_page_token)
        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10)
        
        print("Retrieved points from Qdrant:")
        for point in points:
            print(point)
    except Exception as e:
        print("Error retrieving points from Qdrant:", e)

def retrieve_mongodb_documents():
    try:
        # Connect to MongoDB (assumes it's running on localhost:27017)
        mongo_client = MongoClient("mongodb://localhost:27017/")
        test_db = mongo_client["product_db"]  # Use your test database name
        test_collection = test_db["product_metadata"]  # Use your test collection name

        documents = list(test_collection.find({}))
        print("Retrieved documents from MongoDB:")
        for doc in documents:
            print(doc)
    except Exception as e:
        print("Error retrieving documents from MongoDB:", e)

