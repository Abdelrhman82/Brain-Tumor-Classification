from pathlib import Path

FRAMEWORK = "tensorflow"

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model_T01.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_SIZE = 224