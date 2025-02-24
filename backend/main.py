from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image en bytes
    image_bytes = await file.read()
    
    # Convertir en objet PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Convertir en NumPy array pour un modèle de deep learning
    image_array = np.array(image)

    # Simuler une prédiction (ex: retourner la forme de l'image)
    return {
        "filename": file.filename,
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "shape": image_array.shape
    }
