<<<<<<< HEAD
FROM python:3.11
=======
FROM python:3.9
>>>>>>> 335de9d6e8cfa052357e5dd598e5354d837fae10

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow psycopg2-binary

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
