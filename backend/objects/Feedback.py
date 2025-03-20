import os
import requests
import psycopg2

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
    conn.commit()
    conn.close()

def trigger_github_action():
    token = os.getenv("GITHUB_TOKEN")
    repo_owner = os.getenv("GITHUB_OWNER")
    repo_name = os.getenv("GITHUB_REPO")
    workflow_file = os.getenv("WORKFLOW_FILE")

    requests.post(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_file}/dispatches",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        },
        json={"ref": "main"}
    )