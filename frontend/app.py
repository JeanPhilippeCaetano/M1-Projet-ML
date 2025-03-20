import streamlit as st
import requests
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Classification de Fruits",
    page_icon="🍎",
    layout="centered"
)

# Titre et description
st.title("Classification d'images de fruits 🍎🍌🍇")
st.markdown("Téléchargez une image de fruit pour la classifier et donnez votre feedback!")

# Fonction pour réinitialiser la page
def reset_page():
    st.experimental_rerun()

# Créer un placeholder pour stocker la dernière prédiction
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_image_name' not in st.session_state:
    st.session_state.last_image_name = None

# Uploader de fichier
uploaded_file = st.file_uploader(
    "Déposez une image (PNG ou JPG)", 
    type=["png", "jpg", "jpeg"]
)

# Section de prédiction
if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_column_width=True)
    
    # Préparer la requête API
    API_URL = "http://backend:8000/predict"
    
    # Créer un bouton pour lancer la prédiction
    if st.button("Classifier cette image"):
        with st.spinner("Classification en cours..."):
            # Envoyer l'image au backend
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.success("Image classifiée avec succès!")
                
                # Afficher le résultat
                prediction = data["predictions"][0]["label"]
                confidence = data["predictions"][0]["confidence"]
                
                st.subheader("Résultat de la prédiction:")
                st.markdown(f"Prédiction : **{prediction}** avec confiance **{confidence:.2f}**")
                
                # Stocker la prédiction pour le feedback
                st.session_state.last_prediction = prediction
                st.session_state.last_image_name = data.get("filename", uploaded_file.name)
            else:
                st.error(f"Erreur : Impossible de traiter l'image. Code d'erreur : {response.status_code}")

# Section de feedback (apparaît seulement après une prédiction)
if st.session_state.last_prediction:
    st.markdown("---")
    st.subheader("Donnez votre feedback sur cette prédiction")
    
    # Champ pour le commentaire utilisateur
    user_feedback = st.text_area("Commentaire (optionnel)", height=100)
    
    # Conteneur pour les boutons de feedback
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Prédiction correcte", key="positive_feedback"):
            feedback_data = {
                "image_name": st.session_state.last_image_name,
                "predicted_result": st.session_state.last_prediction,
                "user_feedback": user_feedback,
                "is_good": True
            }
            
            try:
                response = requests.post("http://backend:8000/feedback", json=feedback_data)
                if response.status_code == 200:
                    st.balloons()
                    st.success("Merci pour votre feedback positif!")
                    # Réinitialiser après un court délai pour voir le message de succès
                    st.rerun()
                else:
                    st.error(f"Erreur lors de l'envoi du feedback: {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")
    
    with col2:
        if st.button("👎 Prédiction incorrecte", key="negative_feedback"):
            feedback_data = {
                "image_name": st.session_state.last_image_name,
                "predicted_result": st.session_state.last_prediction,
                "user_feedback": user_feedback,
                "is_good": False
            }
            
            try:
                response = requests.post("http://backend:8000/feedback", json=feedback_data)
                if response.status_code == 200:
                    st.warning("Merci pour votre feedback! Nous allons améliorer notre modèle.")
                    # Réinitialiser après un court délai pour voir le message
                    st.rerun()
                else:
                    st.error(f"Erreur lors de l'envoi du feedback: {response.text}")
            except Exception as e:
                st.error(f"Erreur de connexion: {str(e)}")

# Afficher des informations supplémentaires en bas de page
st.markdown("---")
st.caption("Système de classification d'images de fruits avec feedback pour l'amélioration continue du modèle")