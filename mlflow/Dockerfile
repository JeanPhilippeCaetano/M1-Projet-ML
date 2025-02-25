FROM python:3.11

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow psycopg2-binary

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
