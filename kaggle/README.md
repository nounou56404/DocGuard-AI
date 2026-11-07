# Kaggle (GPU)

Entrainement du CNN et du ViT sur GPU Kaggle, puis `dvc push` des artefacts.

1. Notebook : cloner le depot, `pip install -r requirements.txt`, `dvc pull data`.
2. Secrets (Kaggle Secrets, jamais dans le depot) : `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_TRACKING_URI`.
3. Lancer `dvc repro train@cnn` (ou `train@vit`), puis `dvc push`.
