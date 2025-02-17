from fastapi import FastAPI
import psycopg2
import os

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/mydatabase")

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.get("/health")
def health_check():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return {"status": "OK"}
    except:
        return {"status": "Database connection failed"}
