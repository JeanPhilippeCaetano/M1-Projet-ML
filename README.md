# Projet MLOps - Classification de fruits

Ce projet implémente une solution MLOps complète pour la classification de fruit. Il inclut la pipeline de traitement des données, l'API de prédiction, ainsi que la surveillance et la visualisation des métriques.

## Architecture du projet

- **Modèle ML** : CLassification d'images via le model MobileNetV2
- **API** : FastAPI pour servir les prédictions
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
   - Prometheus
   - Grafana
   - node exporter
   - Evidently
   - PGSQL

3. Accéder aux interfaces :
   - API : http://localhost:8000
   - Prometheus : http://localhost:9090
   - Grafana : http://localhost:3000

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

## Arrêt des services

Pour arrêter tous les services :
```bash
docker-compose down
```
