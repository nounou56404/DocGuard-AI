# Architecture MLOps - DocGuard-AI

Detection de documents **scam / non_scam** (vision, OCR, MLOps). Projet de fin de module Deep Learning Avance.
Un depot, un paquet Python (`docguard_ai`), trois approches de modelisation comparees sur les memes donnees et les memes metriques.

## 1. Principes d'ingenierie

1. **Un seul identifiant : `sample_id`**. Jamais de jointure sur autre chose.
2. **Pas de fuite de donnees** : le split se fait par `group_id` (toutes les images d'un meme document restent dans le meme split). Verifie par l'etape `prepare`.
3. **Meme jeu de test, memes metriques** pour les trois approches : accuracy, precision, recall, f1, auc, `recall_scam`.
4. **Toute experience bat d'abord les baselines** (`reports/baselines.json` : classe majoritaire, tirage aleatoire).
5. **Reproductibilite** : code dans Git, donnees et modeles dans DVC, parametres dans `params.yaml`, seed 42.
6. **Pas de donnees lourdes ni de secrets dans Git** : `.env` ignore, `.env.example` sans valeur.
7. **Une etape = une commande** : `python -m docguard_ai.pipeline.run_stage <etape> [--approach cnn|vit|ocr]`, rejouable par `dvc repro`.
8. **Meme pretraitement OpenCV** a l'entrainement et en production (`vision/preprocessing.py`).

## 2. Vue d'ensemble

```text
 data/raw/<label>/*.png            Git : code, configs, petits CSV, *.dvc
        |  DVC                      DVC : donnees, images traitees, modeles
        v                           MLflow : experiences + Model Registry
 1 validate  -> data/interim/quality.json        (schema, audit, doublons)
 2 prepare   -> data/processed/{manifest.csv, images/}   (split par groupe + OpenCV)
 3 baselines -> reports/baselines.json
        |
        +--> train@cnn | train@vit | train@ocr   -> models/<approche>/{checkpoint, predictions_test.csv}
        +--> evaluate@<approche>                 -> reports/<approche>/metrics.json   (MLflow)
        +--> export@<approche>                   -> models/<approche>/{model.onnx, model_card.json}
        v
 deployment/ : FastAPI + ONNX Runtime (Docker)  ->  frontend/ : React (upload, webcam, resultats)
```

## 3. Arborescence

| Chemin | Role |
|---|---|
| `src/docguard_ai/config.py` | chargement `configs/` et `params.yaml` (racine : `DOCGUARD_ROOT` ou dossier courant) |
| `src/docguard_ai/contract/schemas.py` | schemas Pydantic de l'API (`PredictResponse`, `OcrBox`...) - **contrat gele** |
| `src/docguard_ai/data/` | schema du manifeste, ingestion, audit qualite, split par groupe |
| `src/docguard_ai/vision/` | pretraitement OpenCV, augmentations, explicabilite (Grad-CAM / attention) |
| `src/docguard_ai/ocr/` | moteur OCR, detection des mots suspects |
| `src/docguard_ai/models/` | `cnn.py` (A), `vit.py` (B), `ocr_text.py` (C), baselines, train, evaluate, export ONNX |
| `src/docguard_ai/tracking/` | enveloppe MLflow (conventions de nommage) |
| `src/docguard_ai/pipeline/run_stage.py` | point d'entree des etapes DVC |
| `deployment/` | API FastAPI asynchrone, inference ONNX Runtime, Dockerfile |
| `frontend/` | React + Nginx |
| `configs/`, `params.yaml`, `dvc.yaml` | configuration, hyperparametres, pipeline |
| `tests/` | tests (mini-jeu d'images genere a la volee) |
| `docker-compose.yml` | `api_cnn` 8001, `api_vit` 8002, `api_ocr` 8003, `front` 3000, `mlflow` 5000 |

## 4. Repartition (3 personnes)

| | Personne A | Personne B | Personne C |
|---|---|---|---|
| Approche | CNN, transfer learning, fine-tuning multi-echelles | Vision Transformer (ViT / Swin) | OCR puis classification du texte |
| Fichier | `models/cnn.py` | `models/vit.py` | `models/ocr_text.py` + `ocr/engine.py` |
| Explicabilite | Grad-CAM | carte d'attention | boites OCR + mots suspects |
| Registry MLflow | `scam-cnn` (experience `scam-a`) | `scam-vit` (`scam-b`) | `scam-ocr` (`scam-c`) |
| Service | `api_cnn` (8001) | `api_vit` (8002) | `api_ocr` (8003) |

Le code commun (`data/`, `vision/preprocessing.py`, `contract/`, `models/evaluate.py`, `tracking/`, `pipeline/`) est modifie **seulement avec l'accord des trois**.

## 5. Contrat entre l'entrainement et le reste

- `train(params, config)` ecrit `models/<approche>/checkpoint` et `models/<approche>/predictions_test.csv` (`sample_id, y_true, score`).
- `export(params, config)` ecrit `model.onnx` + `model_card.json` (`name, version, approach, input_size, mean, std, labels, data_version`) apres `check_equivalence`.
- API : `POST /predict?model=cnn|vit|ocr` (image multipart) -> `{label, score, model, heatmap?, ocr?}` ; `GET /health` ; `GET /models`.

## 6. MLflow

| Element | Convention |
|---|---|
| Experiences | `scam-a`, `scam-b`, `scam-c` |
| Nom d'un run | `<approche>-<variante>-<MMJJ>` (ex. `cnn-resnet50-aug-1106`) |
| Metriques | `accuracy, precision, recall, f1, auc, recall_scam` |
| Parametres | `lr, batch_size, epochs, img_size, augmentation, data_version, seed` |
| Artefacts | courbe ROC, matrice de confusion, `model.onnx`, `model_card.json` |
| Registry | `scam-cnn`, `scam-vit`, `scam-ocr` ; Staging puis Production |

## 7. Git et CI

- Branches : `main` (protegee), `feat/cnn`, `feat/vit`, `feat/ocr` ; fusion par pull request relue par une autre personne.
- CI (`.github/workflows/ci.yaml`) : `ruff` -> `pytest` (API, pretraitement, split, equivalence ONNX) -> build des images Docker.
- Fichiers lourds via DVC, jamais Git ; secrets dans `.env` ou Kaggle Secrets.

## 8. Phases

0 contrat + gel - 1 donnees, EDA, baselines - 2 entrainement, comparaison MLflow, Registry - 3 export ONNX + API + tests - 4 React + Docker - 5 CI + README - 6 integration (tableau comparatif, compose complet) + repetition de soutenance.
