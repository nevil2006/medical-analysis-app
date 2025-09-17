import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import torch
import clip
from torchvision import transforms
from model import ClipClassifier, class_names, get_main_type

# -------------------
# Upload folder
# -------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------
# Flask app
# -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------
# Device
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------
# Load CLIP and classifier
# -------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)

classifier = ClipClassifier(clip_model.visual.output_dim, len(class_names)).to(device)

# <-- Correct model path -->
model_path = r"C:\Users\nevil\Desktop\VLM RESEARCH\best_clip_model.pth"
classifier.load_state_dict(torch.load(model_path, map_location=device))
classifier.eval()

# -------------------
# Feature extraction
# -------------------
def extract_features(image_tensor):
    with torch.no_grad():
        features = clip_model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float()
    return features

# -------------------
# Routes
# -------------------
@app.route('/')
def index():
    return render_template('index.html')  # make sure you have index.html in templates/

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    image = Image.open(filename).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = extract_features(image_tensor)
        outputs = classifier(features)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

        pred_subtype = class_names[pred.item()]
        pred_main_type = get_main_type(pred_subtype)

    return jsonify({
        'image_url': '/uploads/' + file.filename,
        'pred_subtype': pred_subtype,
        'pred_main_type': pred_main_type,
        'confidence': round(confidence.item() * 100, 2)  # %
    })

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -------------------
# Run app
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
