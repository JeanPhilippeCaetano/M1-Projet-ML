import streamlit as st
import requests
from PIL import Image
import io 

st.title("Classification d'image de fruits 🍎🍌🍇")

uploaded_file = st.file_uploader(
    "Dépose une image (PNG ou JPG)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Convertir l'image en bytes pour l'envoi
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Envoi à l'API (à adapter selon ton API)
    API_URL = "http://backend:8000/predict"  # Remplace par ton URL d'API
    files = {"file": (uploaded_file.name, img_bytes, "image/jpeg")}
    response = requests.post(API_URL, files=files)

    # if response.status_code == 200:
    #     result = response.json()
    #     st.success(f"Prédiction : {result['class']} (Confiance : {result['confidence']:.2f}%)")
    # else:
    #     st.error("Erreur lors de la prédiction")
