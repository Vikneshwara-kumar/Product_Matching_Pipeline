"""
mongodb_logger.py
-----------------
A module for logging errors and system events to a mocked MongoDB instance.
This module uses Motor (the asynchronous MongoDB driver) to log events.
"""

import asyncio
import logging
import motor.motor_asyncio

# Configure logging for fallback in case MongoDB logging fails.
logging.basicConfig(level=logging.INFO)

# Configuration parameters for the MongoDB logging instance.
LOG_MONGO_URI = "mongodb://localhost:27018"  # Use a separate port/instance for logging.
LOG_DB_NAME = "log_db"
LOG_COLLECTION_NAME = "system_logs"

# Create an asynchronous MongoDB client using Motor.
client = motor.motor_asyncio.AsyncIOMotorClient(LOG_MONGO_URI)
db = client[LOG_DB_NAME]
collection = db[LOG_COLLECTION_NAME]

async def log_event(level: str, message: str, extra: dict = None):
    """
    Asynchronously logs an event to the MongoDB logging collection.

    Args:
        level (str): The log level (e.g., "INFO", "ERROR").
        message (str): The log message.
        extra (dict, optional): Additional contextual data to log.

    Returns:
        The inserted document's ID if successful, or None if logging failed.
    """
    log_doc = {
        "level": level,
        "message": message,
        "extra": extra or {},
    }
    try:
        result = await collection.insert_one(log_doc)
        return result.inserted_id
    except Exception as e:
        # Fallback logging if MongoDB insertion fails.
        logging.error(f"Failed to log event to MongoDB: {e}")
        return None

def log_event_sync(level: str, message: str, extra: dict = None):
    """
    Synchronous wrapper for log_event, allowing logging from non-async code.

    Args:
        level (str): The log level.
        message (str): The log message.
        extra (dict, optional): Additional contextual data.

    Returns:
        The inserted document's ID if successful, or None if logging failed.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If there's no running event loop, create a new one.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(log_event(level, message, extra))
