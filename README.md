# DocGuard-AI - Detection de documents scam / non_scam

Projet de fin de module Deep Learning Avance (vision, OCR, MLOps). Trois approches comparees sur les memes donnees :
**A** CNN (transfer learning, Grad-CAM) - **B** Vision Transformer (attention) - **C** OCR + classification du texte.

Architecture complete : [`docs/architecture.md`](docs/architecture.md).

## Demarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # puis remplir (jamais commite)
pytest -q                       # verifie l'installation
```

## Donnees et pipeline

1. Deposer les images dans `data/raw/scam/` et `data/raw/non_scam/` (licence verifiee, aucune donnee personnelle).
2. Generer le manifeste : `python -c "from docguard_ai.data.ingestion import build_manifest as b; b('data/raw').to_csv('data/metadata/image_manifest.csv', index=False)"`
3. `dvc repro validate prepare baselines`, puis chacun execute `dvc repro train@<approche> evaluate@<approche> export@<approche>`.

## Service

```bash
docker compose up --build       # api_cnn :8001, api_vit :8002, api_ocr :8003, front :3000, mlflow :5000
```
Swagger : `http://localhost:8001/docs`. Les modeles `.onnx` sont montes en lecture seule depuis `models/`.

## Regles d'equipe

- Chacun travaille dans son module (`models/cnn.py`, `vit.py`, `ocr_text.py`) et sa branche `feat/<approche>`.
- Toute modification du code commun exige l'accord des trois. PR relue par au moins une autre personne.
- Pas de fichier lourd ni de secret dans Git (DVC et `.env`).

## Livrables par personne

- [ ] Donnees versionnees DVC, pipeline reproductible
- [ ] >= 3 experiences MLflow comparees, 1 modele dans le Model Registry
- [ ] `model.onnx` + `model_card.json`, equivalence verifiee
- [ ] API FastAPI (`/docs`, Pydantic, tests)
- [ ] Page React (upload, webcam, affichage dynamique)
- [ ] `docker compose up` fonctionnel
- [ ] CI qui passe
- [ ] README : installation, resultats mesures, limites
