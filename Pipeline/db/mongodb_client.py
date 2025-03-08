import motor.motor_asyncio
from utils.logger import log_event_sync  # Import the MongoDB logger

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "product_db"
COLLECTION_NAME = "product_metadata"

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

async def get_product(product_id: str):
    """
    Retrieve product metadata by product id.
    Logs errors to MongoDB if not found or if any exception occurs.
    
    Args:
        product_id (str): The unique identifier for the product.
    
    Returns:
        dict: The product metadata document.
        
    Raises:
        ValueError: If the product is not found.
        Exception: Propagates any other errors after logging.
    """
    try:
        product = await collection.find_one({"_id": product_id})
        if product:
            # Optionally, log the successful retrieval.
            # log_event_sync("INFO", f"Product {product_id} retrieved successfully.", extra={"product_id": product_id})
            return product
        else:
            msg = f"Product with id {product_id} not found."
            log_event_sync("ERROR", msg, extra={"function": "get_product", "product_id": product_id})
            raise ValueError(msg)
    except Exception as e:
        log_event_sync("ERROR", f"Error in get_product: {e}", extra={"function": "get_product", "product_id": product_id})
        raise
