from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = torch.load("model.pth", map_location="cpu")
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return "✅ Pneumonia Detection Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    
    result = "Pneumonia" if pred == 1 else "Normal"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
