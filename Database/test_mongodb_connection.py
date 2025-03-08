# test_mongodb_connection.py
from pymongo import MongoClient

def test_mongodb_connection():
    try:
        # Connect to MongoDB instance (assumes it's running on localhost:27017)
        mongo_client = MongoClient("mongodb://localhost:27017/")
        
        # List available databases
        db_names = mongo_client.list_database_names()
        print("Databases:", db_names)
        
        # Create a test database and collection
        test_db = mongo_client["test_db"]
        test_collection = test_db["test_collection"]
        
        # Insert a test document
        test_doc = {"message": "MongoDB connection successful!"}
        insert_result = test_collection.insert_one(test_doc)
        print("Inserted document ID:", insert_result.inserted_id)
        
        # Retrieve the inserted document
        retrieved_doc = test_collection.find_one({"_id": insert_result.inserted_id})
        print("Retrieved document:", retrieved_doc)
    except Exception as e:
        print("Error testing MongoDB connection:", e)

if __name__ == "__main__":
    test_mongodb_connection()
