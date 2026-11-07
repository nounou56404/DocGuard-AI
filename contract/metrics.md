# Metriques communes (meme jeu de test pour tous)

| Metrique MLflow | Description |
|---|---|
| accuracy | Exactitude globale |
| precision | Precision (classe scam) |
| recall | Rappel (classe scam) |
| f1 | F1 (classe scam) |
| auc | AUC ROC |
| recall_scam | Rappel de la classe scam, suivi a part |

## Conventions MLflow
- Experiences : `scam-a`, `scam-b`, `scam-c`
- Nom de run : `<approche>-<variante>-<date>` (ex. `cnn-resnet50-aug-1106`)
- Parametres : lr, batch_size, epochs, img_size, augmentation, data_version, seed
- Artefacts : courbe ROC, matrice de confusion, model.onnx, model_card.json
- Registry : `scam-cnn`, `scam-vit`, `scam-ocr` (Staging puis Production)
