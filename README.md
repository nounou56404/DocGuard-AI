# DocGuard-AI - Detection de documents scam / non_scam

Projet de fin de module Deep Learning Avance (vision, OCR, MLOps).
Chaque membre livre un systeme complet avec sa propre approche ; les trois suivent le meme contrat.

| Dossier | Approche | Modele Registry | API | Front |
|---|---|---|---|---|
| `a/` | CNN + transfer learning (Grad-CAM) | scam-cnn | 8001 | 3001 |
| `b/` | Vision Transformer (attention) | scam-vit | 8002 | 3002 |
| `c/` | OCR + texte (boites OCR) | scam-ocr | 8003 | 3003 |

## Chaine MLOps
Donnees (DVC) -> Entrainement (MLflow) -> Registry (Staging/Production) -> Export ONNX -> API FastAPI -> Front React -> Docker Compose -> GitHub Actions

## Arborescence
- `contract/` : lecture seule, gele apres le jour 1 (schemas Pydantic, metriques, splits, modele de workflow)
- `a/ b/ c/` : meme structure interne (`data/ src/ api/ frontend/ models/ tests/ dvc.yaml params.yaml docker-compose.yml`)
- `.github/workflows/` : `ci-a.yml`, `ci-b.yml`, `ci-c.yml`
- `docker-compose.yml` : integration finale (phase 6)

## Regles
- Branches : `main` (protegee), `feat/a`, `feat/b`, `feat/c` ; fusion par PR relue par au moins une autre personne.
- Chacun ne modifie que son dossier ; toute modification de `contract/` exige l'accord des trois.
- Aucun import de code entre `a/`, `b/`, `c/` (seul `contract/` est partage).
- Fichiers lourds via DVC, jamais Git. Secrets dans `.env` (jamais commite).

## Contrat API
`POST /predict` (multipart) -> `{label, score, model, heatmap?, ocr?}` ; `GET /health` ; `GET /models`.

## Livrables par personne
- [ ] Donnees versionnees DVC + pipeline reproductible (`dvc repro`)
- [ ] >= 3 experiences MLflow comparees + 1 modele dans le Registry
- [ ] `model.onnx` + `model_card.json` + verification d'equivalence
- [ ] API FastAPI (`/docs`, Pydantic, tests)
- [ ] Page React (upload, webcam, affichage dynamique)
- [ ] `docker-compose up` fonctionnel
- [ ] Workflow GitHub Actions qui passe
- [ ] README (installation, resultats, limites)

## Phases
0 Contrat + gel | 1 Donnees + baseline | 2 Entrainement + Registry | 3 ONNX + API | 4 React + Docker | 5 CI + docs | 6 Integration + soutenance
