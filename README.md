# Projet MLOps - Classification de fruits

Ce projet implémente une solution MLOps complète pour la classification de fruit. 
Il inclut le pipeline de traitement des données, l'API de prédiction, l'utilisation de Mlflow,  ainsi que la surveillance et la visualisation des différentes métriques.

## Architecture du projet

- **Modèle ML** : Classification d'images via le modèle MobileNetV2
- **Frontend** : Interface de l'application
- **Mlflow** : Entraînement et sauvegarde du modèle
- **API (backend)** : FastAPI pour servir les prédictions
- **Monitoring** : Prometheus pour la collecte de métriques + node exporter et evidently
- **Visualisation** : Dashboards Grafana
- **Conteneurisation** : Docker et docker-compose

## Prérequis

- Docker et docker-compose
- Git

## Installation et démarrage

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/JeanPhilippeCaetano/M1-Projet-ML
   cd M1-Projet-ML
   ```

2. Lancer les services :
   ```bash
   docker-compose up -d
   ```

   Cette commande démarre tous les services nécessaires :
   - FASTAPI
   - Streamlit
   - Prometheus
   - Grafana
   - Node exporter
   - Evidently
   - Mlflow
   - PGSQL

3. Accéder aux interfaces :
   - Streamlit (frontend) : http://localhost:8501
   - Fastapi (backend) : http://localhost:8000
   - Prometheus : http://localhost:9090
   - Grafana : http://localhost:3200
   - Mlflow : http://localhost:5000

## Configuration de Grafana

1. Se connecter à Grafana (identifiants par défaut : admin/admin)
2. Ajouter Prometheus comme source de données :
   - Aller dans Configuration > Data Sources
   - Ajouter une source de données
   - Sélectionner Prometheus
   - URL : http://prometheus:9090
   - Sauvegarder et tester

3. Importer les dashboards :
   - Aller dans Dashboard > Import
   - Copier-coller le contenu des fichiers JSON du répertoire "grafana"
   - Sélectionner la source de données Prometheus précédemment configurée

## Utilisation

L'API expose plusieurs endpoints pour effectuer des prédictions sur vos images de fruits.
Vous pouvez utiliser l'interface web streamlit, déposer une image png d'un fruit. Ensuite il faudra cliquer sur 'classifier' et la prédiction s'affichera. 
N'hésitez pas à donner un feedback si jamais la prédiction n'est pas bonne ! Au bout de 5 retours, le modèle pourra se réentraîner automatiquement.

## Arrêt des services

Pour arrêter tous les services :
```bash
docker-compose down
```
