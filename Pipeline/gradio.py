import streamlit as st
from PIL import Image

st.title("Product Matching Pipeline")

st.header("Upload an Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

st.header("Or Enter Text")
user_text = st.text_input("Enter product description or search text:")

# Output area for results
result_area = st.empty()

if st.button("Match Product"):
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Placeholder: Here you would call the image-based product matching function.
        result_area.write("Processing image and fetching best match from MongoDB...\n\n**Best Match:** [Product Metadata Placeholder]")
    elif user_text:
        # Placeholder: Here you would call the text-based product matching function.
        result_area.write("Processing text input and fetching best match from MongoDB...\n\n**Best Match:** [Product Metadata Placeholder]")
    else:
        st.warning("Please upload an image or enter text for matching.")
