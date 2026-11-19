from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import (
    FRAMEWORK,
    MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE
)

# Keep the loaded model available to all requests.
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model_path = Path(MODEL_PATH)

    if not model_path.is_file():
        raise RuntimeError(
            f"Model file not found: {model_path.resolve()}"
        )

    # CPU-only inference.
    with tf.device("/CPU:0"):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    yield

    model = None


app = FastAPI(
    title="Brain MRI Classification API",
    description="Brain MRI classification using TensorFlow MobileNetV2",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "framework": FRAMEWORK,
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = perf_counter()

    # Validate the uploaded file's declared MIME type.
    allowed_content_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are allowed."
        )

    # Read and validate the actual image contents.
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG and PNG images are allowed."
                )

            image = image.convert("RGB")
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
            image_array = np.asarray(
                image,
                dtype=np.float32
            )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    # IMPORTANT:
    # Apply the same normalization used during training.
    #
    # For example, if training used
    # tf.keras.applications.mobilenet_v2.preprocess_input:
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    try:
        with tf.device("/CPU:0"):
            predictions = model(
                image_array,
                training=False
            ).numpy()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        ) from exc

    scores = predictions[0]

    if len(scores) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail="Model output size does not match CLASS_NAMES."
        )

    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_NAMES[predicted_index]

    probabilities = {
        class_name: round(float(score), 6)
        for class_name, score in zip(CLASS_NAMES, scores)
    }

    latency_ms = round(
        (perf_counter() - start_time) * 1000,
        2
    )

    return {
        "framework": FRAMEWORK,
        "label": predicted_label,
        "confidence": round(
            float(scores[predicted_index]), 6
        ),
        "probabilities": probabilities,
        "latency_ms": latency_ms
    }