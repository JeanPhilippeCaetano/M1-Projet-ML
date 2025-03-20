import pandas as pd
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

def save_feedback(filename, true_label):
    """
    Sauvegarder le feedback des utilisateurs dans un fichier CSV et un dossier d'images.
    """
    feedback_dir = "feedback/"
    os.makedirs(feedback_dir, exist_ok=True)

    # Sauvegarde de l'image dans le dossier feedback
    feedback_image_path = os.path.join(feedback_dir, filename)
    with open(feedback_image_path, "wb") as f:
        f.write(filename)

    # Ajouter une ligne dans le CSV de feedbacks
    feedback_csv = os.path.join(feedback_dir, "labels.csv")
    if os.path.exists(feedback_csv):
        df = pd.read_csv(feedback_csv)
    else:
        df = pd.DataFrame(columns=["filename", "label"])
    
    df = df.append({"filename": filename, "label": true_label}, ignore_index=True)
    df.to_csv(feedback_csv, index=False)


def retrain_model():
    """
    Fonction pour réentraîner le modèle avec les images mal classées récupérées dans le dossier feedback.
    """
    # Charger les données de feedback
    feedback_csv = "feedback/labels.csv"
    df = pd.read_csv(feedback_csv)

    # Préparer les données d'entraînement
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory="feedback/",
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical"
    )

    # Charger le modèle existant ou en créer un nouveau
    model = load_model("models/last_model")

    # Compilation du modèle
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Enregistrement du modèle avec MLflow
    mlflow.tensorflow.autolog()

    # Réentraîner le modèle
    with mlflow.start_run():
        model.fit(train_generator, epochs=5)
        model.save("models/last_model")

    # Sauvegarder le modèle dans MLflow
    mlflow.log_artifact("models/last_model")
