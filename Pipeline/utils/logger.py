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
LOG_MONGO_URI = "mongodb://localhost:27017"  # Use a separate port/instance for logging.
LOG_DB_NAME = "log_db"
LOG_COLLECTION_NAME = "system_logs"

# Create an asynchronous MongoDB client using Motor.
client = motor.motor_asyncio.AsyncIOMotorClient(LOG_MONGO_URI)
db = client[LOG_DB_NAME]
collection = db[LOG_COLLECTION_NAME]

async def log_event(level: str, message: str, extra: dict = None):
    """
    Asynchronously logs an event to MongoDB.
    """
    log_doc = {
        "level": level,
        "message": message,
        "extra": extra or {}
    }
    try:
        result = await collection.insert_one(log_doc)
        return result.inserted_id
    except Exception as e:
        logging.error(f"Failed to log event to MongoDB: {e}")
        return None

def log_event_sync(level: str, message: str, extra: dict = None):
    """
    Synchronous wrapper for logging events.
    If an event loop is already running (e.g., in Streamlit), the logging coroutine is scheduled
    as a fire-and-forget task. Otherwise, a new event loop is created to run the coroutine.
    """
    try:
        # Try to get the current running loop.
        loop = asyncio.get_running_loop()
        # If the loop is running, schedule the log_event coroutine.
        if loop.is_running():
            loop.create_task(log_event(level, message, extra))
        else:
            # Should rarely happen, but run synchronously if the loop isn't running.
            loop.run_until_complete(log_event(level, message, extra))
    except RuntimeError:
        # No running loop; create a new one.
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(log_event(level, message, extra))
