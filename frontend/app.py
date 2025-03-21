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
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

    API_URL = "http://backend:8000/predict"
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()
        st.subheader("Résultat de la prédiction :")
        prediction = data["predictions"][0]["label"]
        confidence = data["predictions"][0]["confidence"]
        st.write(f"Prédiction : **{prediction}** avec confiance **{confidence:.2f}**")

        # Boutons de feedback
        if st.button("👍 Correct"):
            requests.post(f"http://backend:8000/feedback", json={"label": prediction, "feedback": "positive"})
            st.success("Merci pour votre feedback !")

        if st.button("👎 Incorrect"):
            requests.post(f"http://backend:8000/feedback", json={"label": prediction, "feedback": "negative"})
            st.warning("Feedback enregistré, merci !")
    else:
        st.error("Erreur : Impossible de traiter l'image")
