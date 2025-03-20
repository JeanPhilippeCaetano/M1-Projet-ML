import os
import requests
import psycopg2
from objects.Feedback import increment_negative_feedback, reset_negative_feedback, save_feedback
import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
from prometheus_client import Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pydantic import BaseModel
import pandas as pd
import json
import tensorflow as tf

app = FastAPI()

# Définition du modèle de feedback
class Feedback(BaseModel):
    image_name: str
    predicted_result: str
    user_feedback: str
    is_good: bool  # Type de feedback (positif/négatif)

# Connexion à MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_latest_model():
    """ Charge dynamiquement le dernier modèle MobileNetV2 enregistré dans MLflow """
    try:
        model_uri = f"models:/mobilenet_v2/Production"  # On récupère la version validée en "Production"
        print(f"🔄 Chargement du modèle depuis MLflow : {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Charger le dernier modèle validé dans MLflow
model = tf.keras.applications.MobileNetV2(weights="imagenet")


# Métrique Prometheus
model_metric = Gauge('model_predictions', 'Nombre de prédictions du modèle')

# Historique des prédictions pour Evidently
prediction_history = []

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        return {"error": "Modèle MLflow non disponible"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prédiction avec le modèle MLflow
    predictions = model.predict(image_array)
    predicted_label = predictions.argmax(axis=1)[0]

    model_metric.inc()

    # Sauvegarder les prédictions pour Evidently
    prediction_history.append({"label": str(predicted_label), "confidence": float(predictions.max())})

    return {
        "filename": file.filename,
        "predictions": [{"label": str(predicted_label), "confidence": float(predictions.max())}]
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

@app.post("/refresh-model")
async def refresh_model():
    """ Recharge le modèle depuis MLflow en cas de mise à jour """
    global model
    model = load_latest_model()

    if model:
        return {"message": "Modèle MLflow mis à jour avec succès"}
    else:
        return {"error": "Échec du chargement du modèle MLflow"}

@app.post("/feedback")
async def feedback(feedback_data: Feedback):
    image_name = feedback_data.image_name
    predicted_result = feedback_data.predicted_result
    user_feedback = feedback_data.user_feedback
    is_good = feedback_data.is_good

    # Enregistrement du feedback dans la DB
    save_feedback(image_name, predicted_result, user_feedback, is_good)

    # Si feedback négatif, on incrémente le compteur
    if not is_good:
        negative_count = increment_negative_feedback()

        if negative_count >= 5:
            reset_negative_feedback()
            return {"message": "Feedback enregistré. Retraining déclenché."}

    return {"message": "Feedback enregistré."}