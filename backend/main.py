from fastapi import FastAPI, File, UploadFile
import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from prometheus_client import start_http_server, Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import json
import subprocess

app = FastAPI()

# Charger le modèle MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Métrique Prometheus
model_metric = Gauge('model_predictions', 'Nombre de prédictions du modèle')

# Historique des prédictions pour Evidently
prediction_history = []

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prédiction
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    model_metric.inc()

    # Sauvegarder les prédictions pour Evidently
    prediction_history.append({"label": decoded_predictions[0][1], "confidence": float(decoded_predictions[0][2])})

    return {
        "filename": file.filename,
        "predictions": [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]
    }

@app.get("/drift-report")
async def drift_report():
    if len(prediction_history) < 3:
        return {"message": "Pas assez de données pour détecter un drift"}

    # Convertir en DataFrame
    df = pd.DataFrame(prediction_history)
    
    # Rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    reference_data = df.iloc[:len(df) // 2]
    current_data = df.iloc[len(df) // 2:] 

    report.run(reference_data=reference_data, current_data=current_data)

    return json.loads(report.json())

@app.post("/feedback")
async def feedback(feedback_data: dict):
    label = feedback_data["label"]
    feedback_type = feedback_data["feedback"]

    if feedback_type == "negative":
        subprocess.run(["gitlab-runner", "exec", "docker", "retrain_model"])
    
    return {"message": "Feedback enregistré"}
