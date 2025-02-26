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
    st.image(image, caption="Image téléchargée", use_container_width=True)

    # Envoyer l'image au backend
    API_URL = "http://backend:8000/predict"  # URL backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        st.subheader("Résultat de la prédiction :")
        for pred in data["predictions"]:
            st.write(f"prédiction de fruits : **{pred['label']}**")
    else:
        st.error("Erreur : Impossible de traiter l'image")
