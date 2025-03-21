from prometheus_client import Gauge, generate_latest
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import io
import json
import subprocess
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import start_http_server, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator


# Charger le modèle MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Prometheus Metrics
model_metric = Gauge('model_predictions', 'Nombre de prédictions du modèle')
drift_metric = Gauge('dataset_drift', 'Indice de drift des données')

# Historique des prédictions
prediction_history = []

app = FastAPI()

# Instrumenter l'application FastAPI avec Prometheus
instrumentator = Instrumentator()
instrumentator.add(app)

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
    print(f"[DEBUG] New prediction: {prediction_history[-1]}")
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
    report_json = json.loads(report.json())

    # Récupérer les résultats du drift
    drift_result = report_json['metrics'][0]['result']['dataset_drift']

    # Exposer la métrique de drift dans Prometheus
    drift_metric.set(drift_result)

    return report_json

# Exposer les métriques Prometheus via un endpoint dédié
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # Cette route permet à Prometheus de récupérer les métriques au format texte brut
    return generate_latest()