import streamlit as st
import asyncio
from PIL import Image
from clip_inference import get_clip_visual_embedding, get_clip_text_embedding
from product_matching import match_product_by_visual, match_product_by_text
from utils.logger import log_event_sync  # logger for MongoDB

st.title("Product Matching Pipeline")
st.write("Upload an image or enter a product description to match a product.")

# Input areas: Image uploader and text input.
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
input_text = st.text_input("Or enter a product description:")

if st.button("Match Product"):
    try:
        # Visual matching if an image is provided.
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            st.info("Extracting visual embedding...")
            visual_embedding = asyncio.run(get_clip_visual_embedding(image))
            
            st.info("Performing product matching on visual data...")
            product = asyncio.run(match_product_by_visual(visual_embedding))
            
            st.success("Product matched successfully!")
            st.write("Matched Product Metadata:")
            st.json(product)
            
            # Log the execution result in MongoDB.
            log_event_sync(
                "RESULT",
                "Visual matching execution result stored.",
                extra={"match_type": "visual", "product": product}
            )
        
        # Text matching if text input is provided.
        elif input_text:
            st.write("Input Text:", input_text)
            
            st.info("Extracting text embedding...")
            text_embedding = asyncio.run(get_clip_text_embedding(input_text))
            
            st.info("Performing product matching on text data...")
            product = asyncio.run(match_product_by_text(text_embedding))
            
            st.success("Product matched successfully!")
            st.write("Matched Product Metadata:")
            st.json(product)
            
            # Log the execution result in MongoDB.
            log_event_sync(
                "RESULT",
                "Text matching execution result stored.",
                extra={"match_type": "text", "product": product, "input_text": input_text}
            )
        else:
            st.warning("Please upload an image or enter text for product matching.")
    
    except Exception as e:
        st.error(f"An error occurred during product matching: {e}")
        log_event_sync(
            "ERROR",
            f"Product matching error: {e}",
            extra={"input_image": bool(uploaded_image), "input_text": input_text}
        )
