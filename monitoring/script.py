import time
import json
import pandas as pd
import threading
import random
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import Gauge, start_http_server

# Simuler une source dynamique (historique des prédictions)
prediction_history = []

# Initialiser Prometheus
drift_gauge = Gauge('dataset_drift', 'Indice de drift des données')

def generate_fake_predictions():
    """ Génère des données factices pour avoir des métriques dans Grafana """
    return {"label": f"Classe_{random.randint(1, 5)}", "confidence": round(random.uniform(0.5, 1.0), 2)}

def compute_drift():
    """ Fonction qui surveille la dérive des données en arrière-plan """
    while True:
        if len(prediction_history) < 10:  # Si pas assez de données, on génère des valeurs factices
            print("[INFO] Pas assez de données, ajout de prédictions factices...")
            for _ in range(5):  
                prediction_history.append(generate_fake_predictions())
            time.sleep(5)
            continue

        df = pd.DataFrame(prediction_history)

        if len(df) < 4:
            print("[WARNING] Pas assez d'échantillons pour analyser le drift")
            time.sleep(10)
            continue

        reference = df.iloc[:len(df) // 2]
        current = df.iloc[len(df) // 2:]

        # Générer le rapport Evidently
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        # Extraire la métrique de dérive
        report_json = json.loads(report.json())
        drift_result = report_json['metrics'][0]['result']['dataset_drift']

        # Mettre à jour Prometheus
        drift_gauge.set(drift_result)

        print(f"[METRICS] Drift détecté: {drift_result}")

        time.sleep(15)

# Démarrer Prometheus
start_http_server(8010)

# Lancer le thread de monitoring
drift_thread = threading.Thread(target=compute_drift, daemon=True)
drift_thread.start()
