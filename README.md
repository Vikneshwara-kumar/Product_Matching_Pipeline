# Product Matching Pipeline

A complete AI-driven product matching pipeline that uses:

- **Triton Inference Server** (for CLIP model inference)
- **Qdrant** (for vector similarity search)
- **MongoDB** (for product metadata and logging)
- **Streamlit** (for a user-facing web UI)

---

## Table of Contents
1. [Overview](#overview)
2. [Demo](#demo)
3. [Architecture](#architecture)

4. [Key Components](#key-components)
   - [Streamlit App (`app.py`)](#streamlit-app-apppy)
   - [CLIP Inference (`clip_inference.py`)](#clip-inference-clip_inferencepy)
   - [Product Matching (`product_matching.py`)](#product-matching-product_matchingpy)
   - [Qdrant Client (`qdrant_client.py`)](#qdrant-client-qdrant_clientpy)
   - [MongoDB Client (`mongodb_client.py`)](#mongodb-client-mongodb_clientpy)
   - [Logging (`logger.py`)](#logging-loggerpy)
5. [Setup & Installation](#setup--installation)
6. [Running the System](#running-the-system)
7. [Usage Guide](#usage-guide)
8. [sample Data](#sample-data-schema)
9. [Developer Guidelines](#developer-guidelines)
10. [License](#license)

---

## Overview
This repository contains a product matching pipeline designed to match user inputs (images or text descriptions) against a catalog of products stored in MongoDB. It leverages:

- **CLIP** for generating embeddings (visual or textual).
- **Triton Inference Server** for efficient model serving.
- **Qdrant** as the vector database for nearest neighbor search.
- **MongoDB** for storing product metadata and logging.

The system also includes an in-memory caching layer to optimize repeated searches, plus robust error handling and logging to a MongoDB collection.

---

## Demo video 
[Watch the demo video](https://raw.githubusercontent.com/https:/Vikneshwara-kumar/Product_Matching_Pipeline/blob/main/assets/demo_video.mp4)(./assets/demo_video.mp4)
<video width="640" height="360" controls>
  <source src="https://raw.githubusercontent.com/https:/Vikneshwara-kumar/Product_Matching_Pipeline/blob/main/assets/demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Architecture

**User Interface (Streamlit)**
- Users can upload an image or enter text to find matching products.

**CLIP Inference**
- Image inputs are processed by `get_clip_visual_embedding`.
- Text inputs are processed by `get_clip_text_embedding`.

**Vector Search (Qdrant)**
- The resulting embedding is passed to Qdrant for a nearest neighbor search.

**Metadata Retrieval (MongoDB)**
- Once a product ID is found, we retrieve product metadata (e.g., name, price, description, image URL/path) from MongoDB.

**Logging**
- Errors and events are logged to a separate MongoDB collection.

**In-Memory Caching**
- For repeated embeddings, the system returns cached results, reducing latency.

---

## Key Components

### Streamlit App (`app.py`)
**Function:** Provides the front-end interface where users can upload an image or enter text, then initiate a match request.

**Features:**
- Displays uploaded images or user-entered text.
- Calls the inference functions and product matching functions.
- Shows matched product metadata in a two-column layout.
- Logs execution results (success or error) to MongoDB.

### CLIP Inference (`clip_inference.py`)
**Function:** Defines asynchronous functions for obtaining text and visual embeddings using the CLIP model served on Triton Inference Server.

**Details:**
- `get_clip_text_embedding(text_prompt: str)` → returns a text embedding.
- `get_clip_visual_embedding(image: PIL.Image.Image)` → returns a visual embedding.
- Both functions preprocess inputs using Hugging Face’s `CLIPProcessor`/`CLIPTokenizer`, then make an inference call to Triton.

### Product Matching (`product_matching.py`)
**Function:** Orchestrates the search in Qdrant and metadata retrieval from MongoDB.

**Details:**
- `match_product_by_text(text_embedding: np.ndarray)`
- `match_product_by_visual(visual_embedding: np.ndarray)`
- Uses an in-memory cache keyed by MD5 hashes of embeddings to avoid redundant searches.

### Qdrant Client (`qdrant_client.py`)
**Function:** Interacts with Qdrant for vector similarity searches.

**Details:**
- `search_embedding(embedding: np.ndarray, collection: str, top_k: int = 5)`
- Wraps the synchronous Qdrant client calls in `asyncio.to_thread` for concurrency.

### MongoDB Client (`mongodb_client.py`)
**Function:** Fetches product metadata from MongoDB by product ID.

**Details:**
- `get_product(product_id: str)`
- Returns the product document from the `product_metadata` collection.

### Logging (`logger.py`)
**Function:** Logs events to a separate MongoDB instance or collection.

**Details:**
- `log_event(level, message, extra)` (async)
- `log_event_sync(level, message, extra)` (sync wrapper)
- Uses Motor (the async MongoDB driver) for insertions; falls back to Python logging on failure.

---

## Setup & Installation

1. **Clone the Repo**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**
    - Create a virtual environment (optional but recommended):
        bash 
        ```python -m venv venv
        source venv/bin/activate or venv\Scripts\activate # on Windows 
        ```
    - Install required packages:
        ```bash 
        pip install -r requirements.txt
        ```

3. **Run or Install Services**

    -   Triton Inference Server: Make sure you have Triton running with the CLIP models (clip_text and clip_visual).
    -   Qdrant: Start a Qdrant instance (e.g., Docker or local binary).
    -   MongoDB: Start your MongoDB instance(s):
    -   One for product metadata (product_db).
    -   Optionally another for logging (log_db).

4. **Quantize the model** (`/Notebook/Model_quantization.ipynb`)
    The Model Quantization Notebook allows you to optimize your model by reducing its size and improving inference speed. You can choose the precision level that best suits your needs:

    -   FP32 (Full Precision): Retains the highest level of accuracy.
    -   FP16 (Half Precision): Offers a balance between speed and precision, reducing memory usage.
    -   INT8 (Integer Quantization): Provides the most significant reduction in model size and latency, ideal for deployment on resource-constrained devices.

    Usage:
    1.  Open the Notebooks/Model_quantization.ipynb notebook in your preferred Jupyter environment.
    2.  Follow the in-notebook instructions to select your preferred quantization format.
    3.  Execute the notebook cells sequentially to quantize and save the model.
    4.  After quantization save the modle in *model/model_repository/* create similar file structure in the directory.

    *Note: The quantization process may affect model accuracy and performance. It is recommended to test the quantized model to ensure it meets your application requirements.*


## Running the System
1.  **Start Qdrant**
    ```bash
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    ```

2.  **Start MongoDB**
    ```bash
    docker run -p 27017:27017 --name mongodb -d mongo:latest
    ```

3.  **Start Triton Inference Server**
    ```bash
    docker run --gpus all -p8000:8000 -p8001:8001 -p8002:8002 nvcr.io/nvidia/tritonserver:xx.xx-py3 \tritonserver --model-repository=/path/to/your/model/repo
    ```

4.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    The app should be available at http://localhost:8501.

## Usage Guide
1.  **Open the Web UI**

    -   Navigate to http://localhost:8501 in your browser.

2.  **Upload an Image or Enter Text**

    -   Click “Browse files” to upload a product image.
    -   Or enter a product description (e.g., “Men’s black puffer jacket”).

3.  **Click “Match Product”**

    -   The system extracts embeddings (via CLIP) and searches Qdrant.
    -   If a match is found, product metadata is displayed.

4.  **Check the Logs**

    Successful or error events are logged in your MongoDB’s log_db → system_logs collection.

## Sample Data Schema
Below is an example JSON document showing how product metadata can be structured in MongoDB:

```json
[
    {
        "id": "1",

        "SKU": "SKU-SONY-WH1000XM5-BEIGE",

        "name": "Sony WH-1000XM5 Noise Cancelling Wireless Headphones (Beige)",

        "brand": "Sony",

        "category": "Electronics / Audio / Headphones",
        "color": "Beige",

        "price": 399.99,

        "description": "Premium over-ear wireless headphones with industry-leading noise cancellation, up to 30 hours of battery life, and superior sound quality.",

        "text": "Sony WH-1000XM5 Noise Cancelling Wireless Headphones (Beige) by Sony are premium over-ear wireless headphones with industry-leading noise cancellation, offering up to 30 hours of battery life and superior sound quality. Available in Beige, they belong to the Electronics / Audio / Headphones category and are priced at $399.99.",

        "image_path": "/path/to/Sony headphone-4.jpg"
    }
]   
```

## Developer Guidelines
1.  **Code Structure**

    -   Each module has a single responsibility (inference, matching, logging, etc.).
    -   Asynchronous functions handle blocking I/O by using asyncio.to_thread.

2.  **Extending the System**

    -   Additional Models: If you want to add new models to Triton, just create new inference functions similar to get_clip_text_embedding and get_clip_visual_embedding.
    -   New Collections: Qdrant can handle multiple collections for different product categories or data modalities. You can update search_embedding to target the new collections.
    -   Advanced Caching: If you need more robust caching, consider a TTL-based or LRU-based approach to prevent unbounded growth.

3.  **Error Handling**

    -   The code uses consistent try/except blocks and logs errors via log_event_sync.
    -   If you want to handle exceptions differently (e.g., custom error pages), adapt the Streamlit error flow in app.py.

4.  **Performance Tuning**

    -   Batching: For high throughput, implement batching at the Triton or Qdrant level.
    -   Client Reuse: If you have many concurrent requests, ensure clients (Triton, Qdrant) are reused effectively.
    -   Async Logging: Currently, logs are inserted asynchronously to avoid blocking the main thread. For extremely high volumes, consider a dedicated logging pipeline.
