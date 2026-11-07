import json

import pandas as pd

from docguard_ai.data.audit import split_leakage
from docguard_ai.data.schema import validate_manifest
from docguard_ai.pipeline import run_stage


def test_validate_then_prepare(mini_project):
    run_stage.main(["validate"])
    quality = json.loads((mini_project / "data/interim/quality.json").read_text())
    assert quality["n_samples"] == 20 and not quality["missing_files"]

    run_stage.main(["prepare"])
    df = pd.read_csv(mini_project / "data/processed/manifest.csv")
    assert not validate_manifest(df)
    assert set(df["split"]) == {"train", "val", "test"}
    assert not split_leakage(df)
    assert all((mini_project / p).exists() for p in df["image_path"])


def test_baselines_and_evaluate(mini_project):
    run_stage.main(["validate"])
    run_stage.main(["prepare"])
    run_stage.main(["baselines"])
    base = json.loads((mini_project / "reports/baselines.json").read_text())
    assert {"majority", "random"} <= set(base)

    df = pd.read_csv(mini_project / "data/processed/manifest.csv")
    test = df[df["split"] == "test"]
    out = mini_project / "models/cnn"
    out.mkdir(parents=True)
    pd.DataFrame({"sample_id": test["sample_id"],
                  "y_true": (test["label"] == "scam").astype(int),
                  "score": (test["label"] == "scam").astype(float)}).to_csv(
        out / "predictions_test.csv", index=False)
    run_stage.main(["evaluate", "--approach", "cnn"])
    m = json.loads((mini_project / "reports/cnn/metrics.json").read_text())
    assert m["accuracy"] == 1.0 and m["recall_scam"] == 1.0
