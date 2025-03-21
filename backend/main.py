from fastapi.responses import PlainTextResponse
from prometheus_client import Gauge, generate_latest
from datetime import datetime
import os
import requests
import psycopg2
from objects.Feedback import get_negative_feedback_dataframe, increment_negative_feedback, launch_training, reset_negative_feedback, save_feedback
import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
import io
from PIL import Image
from prometheus_client import Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pydantic import BaseModel
import pandas as pd
import json
import tensorflow as tf

import json
CLASS_INDEX_PATH = "/backend/class_indices.json"

# Load class indices for decoding predictions
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    print("[INFO] Custom class mapping loaded.")
else:
    # Fallback to ImageNet labels
    class_indices = None
    index_to_class = None
    print("[INFO] No class_indices.json found. Using ImageNet decode_predictions.")

app = FastAPI()

ARCHIVE_DIR = "/backend/archived_images"
os.makedirs(ARCHIVE_DIR, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Définition du modèle de feedback
class Feedback(BaseModel):
    image_name: str
    predicted_result: str
    user_feedback: str
    is_good: bool  # Type de feedback (positif/négatif)

# Connexion à MLflow
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Réentrainement MobileNet V2")
experiment = mlflow.get_experiment_by_name("Réentrainement MobileNet V2")

def load_latest_model():
    """ Charge dynamiquement le dernier modèle MobileNetV2 enregistré dans MLflow """
    try:
        # Obtenir le dernier run de l'expérience
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.tensorflow.load_model(model_uri)
            print(f"Modèle chargé avec succès depuis MLflow (run_id: {run_id})")
        else:
            raise Exception("Aucun run trouvé dans l'expérience")
            
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis MLflow: {e}")
        # Fallback : charger MobileNetV2 depuis Keras si le modèle MLflow n'est pas disponible
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        print("Modèle MobileNetV2 d'origine chargé depuis Keras")
    
    return model

# Charger le dernier modèle validé dans MLflow
model = load_latest_model()

# Prometheus Metrics
model_metric = Gauge('model_predictions', 'Nombre de prédictions du modèle')
drift_metric = Gauge('dataset_drift', 'Indice de drift des données')

# Historique des prédictions
prediction_history = []

app = FastAPI()

@app.get("/")
async def hello():
    return {"message":"hello world !"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), filename: str = Form(...)):
    global model
    if model is None:
        return {"error": "Modèle MLflow non disponible"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    image_path = os.path.join(ARCHIVE_DIR, filename)
    # Sauvegarder l'image dans le dossier "archived_images" with explicit format
    image.save(image_path)
    # Prédiction avec le modèle MLflow
    predictions = model.predict(image_array)
    print(predictions)
    # decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    model_metric.inc()
    if index_to_class:
        predicted_index = int(np.argmax(predictions[0]))
        predicted_label = index_to_class[predicted_index]
        predicted_confidence = float(predictions[0][predicted_index])
    else:
        # fallback: use ImageNet decoding
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
        predicted_label = decoded[1]
        predicted_confidence = float(decoded[2])

    # Sauvegarder les prédictions pour Evidently
    prediction_history.append({"label": predicted_label, "confidence": predicted_confidence})
    print(f"[DEBUG] New prediction: {prediction_history[-1]}")
    return {
    "filename": filename,
    "predictions": [{
        "label": predicted_label,
        "confidence": predicted_confidence
    }]
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
            launch_training()

            return {"message": "Feedback enregistré. Retraining déclenché."}

    return {"message": "Feedback enregistré. " + image_name}

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # Cette route permet à Prometheus de récupérer les métriques au format texte brut
    return generate_latest()