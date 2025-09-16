# Breast Cancer Classification (Histopathology Images)

This project is a **Flask web app** for classifying breast cancer histopathology images using a fine-tuned **CLIP-based PyTorch model**.  
It predicts both the **cancer subtype (8 classes)** and the **main type (benign/malignant)**.

---

## Project Structure

```
medical-image/
├── app.py               # Flask application
├── model.py             # PyTorch model definition
├── classifier.pth       # Trained model weights
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Frontend (upload + results)
└── static/              # (Optional) CSS, JS, images
```

---

## Installation & Setup

### 1️ Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/medical-image-classification.git
cd medical-image-classification
```

### 2️ Create Environment

**Using conda:**
```bash
conda create -n medclf python=3.9 -y
conda activate medclf
```

**Or using venv:**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️ Run the App

```bash
python app.py
```

---

## Usage

1. Open your browser and go to `http://127.0.0.1:5000/`.
2. Upload a histopathology image.
3. View the predicted cancer subtype and main type.

---

## License

[MIT License](LICENSE)