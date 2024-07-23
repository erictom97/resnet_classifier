import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import io

# Load the pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Load ImageNet class names
with open('imagenet_class_index.json') as f:
    imagenet_class_index = json.load(f)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_bytes):
    # print(type(image_bytes))
    # print(io.BytesIO(image_bytes))
    # image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image_bytes).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    class_id, class_name = imagenet_class_index[str(predicted_idx)]
    return class_id, class_name

# Streamlit app
st.title("Image Classification with ResNet50")
st.write("Upload an image to classify it using a pretrained ResNet50 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    print(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    img_bytes = uploaded_file.read()
    class_id, class_name = get_prediction(image)
    
    # Display the prediction
    st.write(f"Prediction: {class_name}")