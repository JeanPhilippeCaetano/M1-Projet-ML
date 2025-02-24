FROM python:3.9

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow psycopg2-binary

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
