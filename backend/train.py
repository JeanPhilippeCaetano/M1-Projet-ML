from datetime import datetime
import pandas as pd
import psycopg2
from objects.Feedback import get_negative_feedback_dataframe, reset_negative_feedback
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

def main():
    """
    Fonction pour fine-tuner le modèle MobileNetV2 pré-entraîné avec les images mal classées.
    """
    print("Démarrage du fine-tuning...")
    print("Étape 1: Récupération des données de feedback")
    # Charger les données de feedback
    df = get_negative_feedback_dataframe()
    print(df)
    # Vérifier si le DataFrame contient des données
    if df.empty:
        print("Aucune donnée de feedback négatif disponible pour le fine-tuning.")
        return
    
    print("Étape 2: Connexion à MLflow")
    # Connexion à MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Réentrainement MobileNet V2")
    experiment = mlflow.get_experiment_by_name("Réentrainement MobileNet V2")
    
    # Activer l'autolog de TensorFlow pour MLflow
    mlflow.tensorflow.autolog()
    
    # Chemin vers le modèle sauvegardé
    
    try:
        # Obtenir le dernier run de l'expérience
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.tensorflow.load_model(model_uri)
            print(f"Modèle chargé avec succès depuis MLflow (run_id: {run_id})")
        else:
            raise Exception("Aucun run trouvé dans l'expérience")
            
    except Exception as e:
        print(f"Erreur lors du chargement du modèle depuis MLflow: {e}")
        # Fallback : charger MobileNetV2 depuis Keras si le modèle MLflow n'est pas disponible
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        print("Modèle MobileNetV2 d'origine chargé depuis Keras")
    
    print("Étape 4: Préparation du générateur de données")
    # Préparer les données d'entraînement - Utiliser le générateur d'images
    # Ici on suppose que les chemins dans df['filename'] sont corrects
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # After loading the model, but before compiling it:
    # 1. Count your unique classes
    num_classes = len(df['label'].unique())

    # 2. Remove the top layer of MobileNetV2
    base_model = model
    x = base_model.layers[-2].output  # Get the output of the second-to-last layer

    # 3. Add a new classification layer
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    print("Étape 5: Configuration du modèle pour fine-tuning")
    # Créer le générateur à partir du DataFrame
    # Si les chemins sont absolus, utiliser directory=None
    # Si les chemins sont relatifs, spécifier le répertoire de base
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory="backend/archived_images",  # Utiliser None si les chemins dans df['filename'] sont absolus
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=4,  # Petit batch size pour le fine-tuning
        class_mode="categorical"
    )
    
    # Garder les poids des couches préentraînées fixes pour éviter d'oublier
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    # Seules les dernières couches seront réentraînées
    for layer in model.layers[-3:]:
        layer.trainable = True
    
    # Compiler le modèle pour le fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Faible taux d'apprentissage
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Étape 6: Démarrage de l'entraînement")
    # Démarrer un run MLflow pour suivre l'expérience
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name='FineTuning_' + datetime.now().strftime("%Y%m%d_%H%M%S")):
        # Réentraîner le modèle avec peu d'epochs
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, len(df) // 4),  # Au moins 1 step
            epochs=3,  # Peu d'epochs pour le fine-tuning
            verbose=1
        )

        print("Étape 7: Sauvegarde du modèle")     
        # Enregistrer le modèle dans MLflow
        mlflow.tensorflow.log_model(model, "model")
        print("Modèle enregistré dans MLflow")
    
    print("Étape 8: Réinitialisation du compteur de feedback")
    # Réinitialiser le compteur de feedback négatif
    reset_negative_feedback()

    print("Fine-tuning terminé avec succès")

if __name__ == "__main__":
    main()