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

def transform_image(image):
    return preprocess(image).unsqueeze(0)

def get_prediction(image):
    tensor = transform_image(image)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    class_id, class_name = imagenet_class_index[str(predicted_idx)]
    return class_id, class_name

# Streamlit app
st.title("Image Classification with ResNet50")
st.write("Use your camera to classify images in real-time using a pretrained ResNet50 model.")

# Camera input
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    # Display the captured image
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    # Perform prediction
    class_id, class_name = get_prediction(image)
    
    # Display the prediction
    st.write(f"Prediction: {class_name}")

# Optional: Keep the file uploader for comparison
st.write("Or upload an image to classify:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    class_id, class_name = get_prediction(image)
    
    # Display the prediction
    st.write(f"Prediction: {class_name}")