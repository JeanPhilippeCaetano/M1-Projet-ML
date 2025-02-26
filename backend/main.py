from fastapi import FastAPI, File, UploadFile
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from prometheus_client import start_http_server, Gauge

app = FastAPI()

# Charger modèle pré-entraîné 
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Metric pour prometheus
model_metric = Gauge('model_predictions', 'Number of predictions made by the model')

# Imagenet
imagenet_labels = {i: v for i, v in enumerate(tf.keras.applications.mobilenet_v2.decode_predictions(np.expand_dims(np.arange(1000), axis=0), top=1000)[0])}

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image en bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # preprocess
    image = image.resize((224, 224))  # Redimensionner
    image_array = np.array(image) / 255.0  # Normaliser
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch
    # predict
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    model_metric.inc()
    # return
    return {
        "filename": file.filename,
        "predictions": [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]
    }

start_http_server(8001)  # Scraper Prometheus
