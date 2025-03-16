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

    # Convertir l'image en bytes pour l'envoi
    # img_bytes = io.BytesIO()
    # image.save(img_bytes, format="JPEG")
    # img_bytes = img_bytes.getvalue()

    # Envoi à l'API (à adapter selon ton API)
    API_URL = "http://backend:8000/predict"  # Remplace par ton URL d'API
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        st.json(response.json())  # Affiche les infos de l'image
        if st.button('👍 Good prediction'):
            st.write('good')
        elif st.button('👎 Bad prediction'):
            st.write('bad')
            true_response = st.text_input('Correction : ')
    else:
        st.error("Error: Unable to process the image")
        
