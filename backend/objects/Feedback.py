import os
import subprocess
import requests
import psycopg2
import pandas as pd

def connect_db():
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@db:5432/mydatabase')
    return psycopg2.connect(DATABASE_URL)

def save_feedback(image_name, predicted_result, user_feedback, is_good):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback (image_name, predicted_result, user_feedback, is_good) 
        VALUES (%s, %s, %s, %s);
    """, (image_name, predicted_result, user_feedback, is_good))
    conn.commit()
    conn.close()

def increment_negative_feedback():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE feedback_counter 
        SET negative_count = negative_count + 1 
        RETURNING negative_count;
    """)
    count = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return count

def reset_negative_feedback():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("UPDATE feedback_counter SET negative_count = 0;")
    cur.execute("TRUNCATE TABLE feedback;")
    conn.commit()
    conn.close()

def launch_training():
    """ Exécute le script train.py en arrière-plan """
    script_path = os.path.join(os.getcwd(), "train.py")  # Assure-toi que le fichier est bien à la racine du projet
    subprocess.Popen(["python", script_path])


def get_negative_feedback_dataframe():
    try:
        # Création d'un curseur pour exécuter la requête SQL
        conn = connect_db()
        cursor = conn.cursor()
        
        # Requête SQL pour récupérer uniquement les feedbacks négatifs
        query = """
        SELECT image_name, user_feedback
        FROM feedback
        WHERE is_good = False
        ORDER BY created_at DESC
        """
        
        cursor.execute(query)
        
        # Récupération des résultats
        results = cursor.fetchall()
        # Création du DataFrame avec les colonnes 'filename' et 'label'
        df = pd.DataFrame(results, columns=['filename', 'label'])
        
        cursor.close()
        
        return df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données de feedback négatif : {e}")
        return pd.DataFrame(columns=['filename', 'label'])  # Retourne un DataFrame vide en cas d'erreur