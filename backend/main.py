from prometheus_client import Gauge, generate_latest
from fastapi import FastAPI, File, UploadFile
import io
import json
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import start_http_server
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

def generate_fake_predictions():
    """ Génère des données factices pour tester Grafana """
    return {"label": f"Classe_{random.randint(1, 5)}", "confidence": round(random.uniform(0.5, 1.0), 2)}

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
    
    print(f"[METRICS] Nouvelle prédiction ajoutée: {prediction_history[-1]}")

    return {
        "filename": file.filename,
        "predictions": [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]
    }

@app.get("/drift-report")
async def drift_report():
    if len(prediction_history) < 3:
        print("[WARNING] Pas assez de données pour détecter un drift")
        return {"message": "Pas assez de données pour détecter un drift"}

    df = pd.DataFrame(prediction_history)
    
    # Rapport Evidently
    report = Report(metrics=[DataDriftPreset()])
    reference_data = df.iloc[:len(df) // 2]
    current_data = df.iloc[len(df) // 2:] 

    print(f"[INFO] Données de référence: \n{reference_data.head()}")
    print(f"[INFO] Données actuelles: \n{current_data.head()}")

    report.run(reference_data=reference_data, current_data=current_data)
    report_json = json.loads(report.json())

    # Récupérer le résultat du drift
    drift_result = report_json['metrics'][0]['result']['dataset_drift']

    # Mise à jour Prometheus
    drift_metric.set(drift_result)

    print(f"[METRICS] Drift détecté: {drift_result}")

    return report_json


Instrumentator().instrument(app).expose(app)


@app.get("/metrics")
async def metrics():
    try:
        metrics_data = generate_latest()  # Génère les métriques sous format Prometheus
        return metrics_data
    except Exception as e:
        print(f"Erreur lors de la génération des métriques: {e}")
        return {"error": "Failed to generate metrics"}


# Démarrer Prometheus
start_http_server(8001)
print("[INFO] Serveur Prometheus démarré sur le port 8001")
