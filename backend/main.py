from fastapi import FastAPI, File, UploadFile
from evidently.dashboard import Dashboard
from evidently.metrics import RegressionQualityMetric
import io
import numpy as np
from PIL import Image
from prometheus_client import start_http_server, Gauge

app = FastAPI()

# Initialisation des métriques Prometheus
model_metric = Gauge('model_predictions', 'Number of predictions made by the model')

# Exemple de Dashboard Evidently
dashboard = Dashboard(metrics=[RegressionQualityMetric()])  # a voir en fonction du model

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image en bytes
    image_bytes = await file.read()

    # Convertir en objet PIL
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)

    prediction = image_array.shape  # predict

    # Mettre à jour les métriques Prometheus
    model_metric.inc()

    return {
        "filename": file.filename,
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "shape": image_array.shape
    }


start_http_server(8001)  # port de scrapping metrics
