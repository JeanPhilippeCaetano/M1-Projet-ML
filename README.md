# Projet MLOps - Détection d'intrusion avec KDDCUP99

Ce projet implémente une solution MLOps complète pour la détection d'intrusions réseau en utilisant le dataset KDDCUP99. Il inclut le pipeline de traitement des données, l'entraînement du modèle, l'API de prédiction, ainsi que la surveillance et la visualisation des métriques.

## Architecture du projet

- **Modèle ML** : Détection d'intrusion réseau basée sur KDDCUP99
- **API** : FastAPI pour servir les prédictions
- **Monitoring** : Prometheus pour la collecte de métriques
- **Visualisation** : Dashboards Grafana
- **Conteneurisation** : Docker et docker-compose

## Prérequis

- Docker et docker-compose
- Git

## Installation et démarrage

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-username/mlops-kddcup99.git
   cd mlops-kddcup99
   ```

2. Lancer les services :
   ```bash
   docker-compose up -d
   ```

   Cette commande démarre tous les services nécessaires :
   - API de prédiction
   - Prometheus
   - Grafana
   - Autres services de l'infrastructure

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

L'API expose plusieurs endpoints pour effectuer des prédictions sur des données réseau. Consultez la documentation de l'API disponible à l'adresse http://localhost:8000/docs pour plus de détails.

## Arrêt des services

Pour arrêter tous les services :
```bash
docker-compose down
```
