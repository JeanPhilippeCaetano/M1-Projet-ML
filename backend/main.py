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

app = FastAPI()

ARCHIVE_DIR = "archived_images"
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

# Métrique Prometheus
model_metric = Gauge('model_predictions', 'Nombre de prédictions du modèle')

# Historique des prédictions pour Evidently
prediction_history = []

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

    # Générer un nom de fichier unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}_{file.filename}"

    # Make sure we have a valid extension
    # if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    #     # Default to PNG if no valid extension
    #     filename = f"{filename}.png"
    print("filename : ",filename)
    image_path = os.path.join(ARCHIVE_DIR, filename)
    print("image_path : ", image_path)
    # Sauvegarder l'image dans le dossier "archived_images" with explicit format
    image.save(image_path)
    # Prédiction avec le modèle MLflow
    predictions = model.predict(image_array)
    predicted_label = predictions.argmax(axis=1)[0]

    model_metric.inc()

    # Sauvegarder les prédictions pour Evidently
    prediction_history.append({"label": str(predicted_label), "confidence": float(predictions.max())})

    return {
        "filename": filename,
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
            launch_training()

            return {"message": "Feedback enregistré. Retraining déclenché."}

    return {"message": "Feedback enregistré. " + image_name}