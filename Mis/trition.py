
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype
from PIL import Image
import clip
import torch

# Model settings (update these if needed)
MODEL_NAME = "clip_visual"
MODEL_VERSION = "1"  # or leave empty for the latest version

# Triton server URL (default HTTP port is 8000)
TRITON_URL = "localhost:8000"

# Load CLIP model to get the preprocess function (we only need the preprocess)
device = "cuda"  # Preprocessing can be done on CPU
_, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image_path = "/home/vicky/Product_Matching_Pipeline/Dataset/Adidas-1.jpg"  # Update with your image file path
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image)  # returns a tensor of shape [3, 224, 224] in float32
print("Input tensor shape:", input_tensor.shape)

# Convert the tensor to numpy array and cast to FP16
input_np = input_tensor.numpy().astype(np.float16)
# Add a batch dimension: [1, 3, 224, 224]
input_np = np.expand_dims(input_np, axis=0)

# Create Triton client
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

# Prepare the inference input object (input name must match your model configuration)
infer_input = httpclient.InferInput("INPUT__0", input_np.shape, "FP16")
infer_input.set_data_from_numpy(input_np)

# Prepare the inference output object (output name must match your model configuration)
infer_output = httpclient.InferRequestedOutput("OUTPUT__0")

# Send the inference request
response = triton_client.infer(
    model_name=MODEL_NAME,
    model_version=MODEL_VERSION,
    inputs=[infer_input],
    outputs=[infer_output]
)

# Retrieve and print the output embedding
output_embedding = response.as_numpy("OUTPUT__0")
print("Output embedding shape:", output_embedding.shape)
print("Output embedding:", output_embedding)