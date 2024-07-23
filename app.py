from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import json
import io
import base64

app = Flask(__name__)

# Load the pretrained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Load ImageNet class names
with open('imagenet_class_index.json') as f:
    imagenet_class_index = json.load(f)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform_image(image_bytes):
    print(type(image_bytes))
    print(io.BytesIO(image_bytes))
    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    class_id, class_name = imagenet_class_index[str(predicted_idx)]
    return class_id, class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform prediction
            with open(filepath, 'rb') as f:
                img_bytes = f.read()
            class_id, class_name = get_prediction(img_bytes)
            
            return render_template('result.html', filename=filename, prediction=class_name)
    return render_template('index.html')

@app.route('/classify_frame', methods=['POST'])
def classify_frame():
    data = request.get_json()
    image_data = data['image'].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    class_id, class_name = get_prediction(image_bytes)
    return jsonify({'class_id': class_id, 'class_name': class_name})

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')