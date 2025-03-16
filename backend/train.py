import tensorflow as tf
import mlflow
import mlflow.tensorflow
import os

# Chargement des nouvelles images incorrectes (ex: depuis un dossier feedback/)
# Supposons que feedback/ contient des images mal classifiées avec leur vrai label dans un CSV
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Chargement des nouvelles images
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

# Enregistrement avec MLflow
mlflow.tensorflow.autolog()
with mlflow.start_run():
    model.fit(train_generator, epochs=5)
    model.save("models/last_model")

    mlflow.log_artifact("models/last_model")
