# Modeles

Les modeles ne sont **pas** versionnes dans Git : ils sont suivis par DVC (`dvc.yaml`).

- `models/<approche>/checkpoint` : poids d'entrainement (cnn, vit, ocr)
- `models/<approche>/predictions_test.csv` : scores sur le jeu de test (entree de l'evaluation commune)
- `models/<approche>/model.onnx` + `model_card.json` : livrables pour l'API
- Recuperer les modeles du commit courant : `dvc pull`
- Runs et metriques : MLflow ; metriques finales : `reports/<approche>/metrics.json`
