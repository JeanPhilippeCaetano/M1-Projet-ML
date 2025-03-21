from datetime import datetime
import json
import pandas as pd
import psycopg2
from objects.Feedback import get_negative_feedback_dataframe, reset_negative_feedback
from mlflow.models.signature import infer_signature
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

    # Avant de créer le générateur
    print("Fichiers à traiter:")
    for idx, filename in enumerate(df['filename']):
        print(f"Contenu: {os.listdir('/backend/archived_images')}")
        full_path = os.path.join("/backend/archived_images", filename)
        print(f"{idx+1}. Recherche du fichier: {full_path} (existe: {os.path.exists(full_path)})")


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
        fill_mode='nearest',
        validation_split=0.2  # Ajouter une validation split pour surveiller l'apprentissage
    )
    
    # Vérifier les classes uniques dans df
    unique_classes = df['label'].unique()
    num_classes = len(unique_classes)
    print(f"Classes détectées: {unique_classes}")
    print(f"Nombre de classes: {num_classes}")
    
    # Créer le générateur à partir du DataFrame
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory="/backend/archived_images",
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=4,
        class_mode="categorical",  # S'assurer que c'est en mode catégorique
        shuffle=True
    )
    
    # Vérifier les classes du générateur
    print(f"Mapping des classes du générateur: {train_generator.class_indices}")
    print(f"Forme des données (batch): {next(train_generator)[0].shape}, {next(train_generator)[1].shape}")
    
    # Réinitialiser le générateur après vérification
    train_generator.reset()
    
    # Modifier le modèle pour correspondre au nombre correct de classes
    base_model = model
    
    # Reconstruire le modèle avec le bon nombre de classes en sortie
    if hasattr(base_model, 'layers') and len(base_model.layers) > 1:
        x = base_model.layers[-2].output
    else:
        # Si le modèle n'a pas la structure attendue, prenez la sortie avant la couche de classification
        x = base_model.get_layer('global_average_pooling2d').output
    
    # Créer une nouvelle couche de sortie avec le bon nombre de classes
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    
    print("Étape 5: Configuration du modèle pour fine-tuning")
    # Garder les poids des couches préentraînées fixes pour éviter d'oublier
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    # Seules les dernières couches seront réentraînées
    for layer in model.layers[-3:]:
        layer.trainable = True
    
    # Compiler le modèle pour le fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Vérifier la forme de sortie du modèle
    print(f"Forme de sortie du modèle: {model.output_shape}")
    
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

        # Take a single batch from the generator to infer signature
        sample_batch = next(train_generator)
        input_example = sample_batch[0]
        output_example = model.predict(input_example)

        signature = infer_signature(input_example, output_example)

        print("Étape 7: Sauvegarde du modèle")     
        # Enregistrer le modèle dans MLflow
        class_indices_path = "/backend/class_indices.json"
        with open(class_indices_path, "w") as f:
            json.dump(train_generator.class_indices, f)
        print(f"Mapping des classes sauvegardé dans: {class_indices_path}")

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )        
        print("Modèle enregistré dans MLflow")
    
    print("Étape 8: Réinitialisation du compteur de feedback")
    # Réinitialiser le compteur de feedback négatif
    reset_negative_feedback()

    print("Fine-tuning terminé avec succès")

if __name__ == "__main__":
    main()