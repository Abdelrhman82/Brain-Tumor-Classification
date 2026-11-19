# Brain Tumor MRI Classification & Deployment

A deep learning project for classifying brain MRI scans into four tumor categories (Glioma, Meningioma, Pituitary, or No Tumor) and deploying the trained model via a production-ready **FastAPI** REST API.

---

## 📌 Project Overview

This repository demonstrates the end-to-end machine learning lifecycle for medical image analysis:
- **Exploration & Transfer Learning:** Comparison and training across modern deep learning frameworks (**TensorFlow / Keras** with `MobileNetV2` and **PyTorch** with `VGG11_bn` and Grad-CAM interpretability).
- **High Performance:** The deployed MobileNetV2 model achieves **~91% test accuracy** across four distinct classes.
- **Inference Service:** A lightweight, asynchronous REST API powered by **FastAPI** optimized for CPU inference with input validation and latency benchmarking.

---

## 🧠 Dataset & Classes

The dataset consists of T1-weighted contrast-enhanced brain MRI images split into training and testing sets across four classes:

| Class | Description |
| :--- | :--- |
| **`glioma`** | Tumors originating in the glial cells of the brain or spine |
| **`meningioma`** | Tumors arising from the meninges (membranes surrounding brain & spinal cord) |
| **`notumor`** | Healthy / normal brain scans without detected tumors |
| **`pituitary`** | Tumors affecting the pituitary gland |

---

## 📂 Project Structure

```plaintext
.
├── app/
│   ├── config.py             # Application configuration (model path, image size, classes)
│   └── main.py               # FastAPI application & prediction endpoint
├── model/
│   └── model_T01.keras       # Saved trained MobileNetV2 Keras model
├── notebooks/
│   ├── pytorch_T01.ipynb     # PyTorch exploration (VGG11_bn, training loop, Grad-CAM)
│   └── tensorflow_T01.ipynb  # TensorFlow pipeline (MobileNetV2 training & evaluation)
├── report_T01.pdf            # Comprehensive analysis and project report
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📊 Model Performance (TensorFlow MobileNetV2)

The MobileNetV2 model was fine-tuned on $224 \times 224$ MRI images:

- **Overall Test Accuracy:** `91.0%`
- **Test Loss:** `0.5645`

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 0.95 | 0.72 | 0.82 | 400 |
| **Meningioma** | 0.80 | 0.93 | 0.86 | 400 |
| **No Tumor** | 0.96 | 1.00 | 0.98 | 400 |
| **Pituitary** | 0.96 | 0.99 | 0.98 | 400 |
| **Weighted Avg** | **0.92** | **0.91** | **0.91** | **1600** |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+ (Python 3.10 – 3.12 recommended)
- `pip` or virtual environment manager (`venv` / `conda`)

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 3. Set Up Virtual Environment & Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (cmd):
venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## ⚡ Running the FastAPI Server

Start the API service with `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running:
- **Interactive Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔌 API Endpoints

### 1. Health Check
- **`GET /health`**
- Returns the API status, loaded framework, and supported target classes.

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "framework": "tensorflow",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

### 2. Predict Brain Tumor
- **`POST /predict`**
- Accepts an uploaded MRI image (`image/jpeg` or `image/png`) via `multipart/form-data`.

#### Example via cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/Testing/pituitary/Te-pi_1.jpg"
```

#### Example via Python:
```python
import requests

url = "http://localhost:8000/predict"
file_path = "data/Testing/pituitary/Te-pi_1.jpg"

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": ("image.jpg", f, "image/jpeg")})

print(response.json())
```

**Response Example:**
```json
{
  "framework": "tensorflow",
  "label": "pituitary",
  "confidence": 0.998241,
  "probabilities": {
    "glioma": 0.000120,
    "meningioma": 0.001625,
    "notumor": 0.000014,
    "pituitary": 0.998241
  },
  "latency_ms": 32.45
}
```

---

## 📓 Notebooks

- **`notebooks/tensorflow_T01.ipynb`**: Demonstrates data loading, preprocessing, MobileNetV2 model configuration, transfer learning, evaluation metrics, and saving the model to `.keras` format.
- **`notebooks/pytorch_T01.ipynb`**: Demonstrates PyTorch data pipeline with augmentations, transfer learning using `VGG11_bn`, custom training loop, and model interpretability with **Grad-CAM**.

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
