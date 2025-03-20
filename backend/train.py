import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Connexion à MLflow
mlflow.set_tracking_uri("http://mlflow:5000")  # Assure-toi que l'URI MLflow est correcte
mlflow.set_experiment("MobileNetV2 Retraining")  # Nom de l'expérience MLflow
mlflow.get_experiment_by_name("MobileNetV2 Retraining")# Nom de l'expérience de votre groupe

# Chargement des nouvelles images mal classifiées
df = pd.read_csv("feedback/labels.csv")
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

# Définition du modèle
model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3), classes=len(df['label'].unique()))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Enregistrement automatique des paramètres et métriques dans MLflow
mlflow.tensorflow.autolog()

with mlflow.start_run():
    model.fit(train_generator, epochs=5)

    # Sauvegarde du modèle dans MLflow
    model_path = "models/last_model"
    model.save(model_path)

    # Log du modèle comme un artifact
    mlflow.tensorflow.log_model(tf_model=model, artifact_path="mobilenet_v2_model")

    # Enregistrement du modèle dans MLflow Model Registry
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/mobilenet_v2_model"
    mlflow.register_model(model_uri, "mobilenet_v2")

    print(f"✅ Modèle enregistré sous : {model_uri}")
