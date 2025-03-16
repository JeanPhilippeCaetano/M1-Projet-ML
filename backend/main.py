from fastapi import FastAPI, File, UploadFile
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from prometheus_client import start_http_server, Gauge
import json
import subprocess

app = FastAPI()

model = tf.keras.applications.MobileNetV2(weights="imagenet")
model_metric = Gauge('model_predictions', 'Number of predictions made by the model')

feedback_counter = {}  # Stocke les erreurs pour chaque label

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    model_metric.inc()

    return {
        "filename": file.filename,
        "predictions": [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]
    }

@app.post("/feedback")
async def feedback(feedback_data: dict):
    label = feedback_data["label"]
    feedback_type = feedback_data["feedback"]

    if feedback_type == "negative":
        feedback_counter[label] = feedback_counter.get(label, 0) + 1

        if feedback_counter[label] >= 5:
            trigger_training()

    return {"message": "Feedback enregistré"}

def trigger_training():
    subprocess.run(["gitlab-runner", "exec", "docker", "retrain_model"])
