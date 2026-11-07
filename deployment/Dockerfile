FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY deployment/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY pyproject.toml params.yaml ./
COPY configs ./configs
COPY src ./src
COPY deployment ./deployment
RUN pip install --no-cache-dir -e .
# Les modeles ne sont pas copies dans l'image : ils sont montes en lecture seule (/app/models)
EXPOSE 8000
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
