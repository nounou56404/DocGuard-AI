"""Point d'entree d'une etape : python -m docguard_ai.pipeline.run_stage <etape> [--approach X]."""
from __future__ import annotations

import argparse
import json

SIMPLE_STAGES = ("validate", "prepare", "baselines")
APPROACH_STAGES = ("train", "evaluate", "export")
APPROACHES = ("cnn", "vit", "ocr")


def _validate() -> None:
    import pandas as pd

    from docguard_ai.config import get_config, resolve
    from docguard_ai.data.audit import audit_manifest
    from docguard_ai.data.schema import validate_manifest

    cfg = get_config()
    df = pd.read_csv(resolve(cfg["data"]["manifest_path"]), keep_default_na=False)
    errors = validate_manifest(df)
    if errors:
        raise SystemExit("manifeste invalide : " + "; ".join(errors))
    report = audit_manifest(df)
    out = resolve(cfg["outputs"]["quality"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


def _prepare() -> None:
    import cv2
    import pandas as pd

    from docguard_ai.config import get_config, get_params, project_root, resolve
    from docguard_ai.data.audit import split_leakage
    from docguard_ai.data.split import assign_splits
    from docguard_ai.vision.preprocessing import preprocess_image

    cfg, params = get_config(), get_params()
    df = pd.read_csv(resolve(cfg["data"]["manifest_path"]), keep_default_na=False)
    p, pre = params["prepare"], params["preprocess"]
    if (df["split"] == "").any():          # splits deja fournis par le contrat : on les garde
        df = assign_splits(df, p["group_col"], p["val_size"], p["test_size"], params["seed"])
    leak = split_leakage(df, p["group_col"])
    if leak:
        raise SystemExit(f"fuite entre splits pour les groupes : {leak[:5]}")
    img_dir = resolve(cfg["outputs"]["images_dir"])
    img_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for sid, path in zip(df["sample_id"], df["image_path"]):
        img = cv2.imread(str(resolve(path)))
        out = preprocess_image(img, pre["img_size"], pre["deskew"], pre["denoise"])
        dst = img_dir / f"{sid}.png"
        cv2.imwrite(str(dst), out)
        new_paths.append(str(dst.relative_to(project_root())))
    df["image_path"] = new_paths
    manifest = resolve(cfg["outputs"]["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=SIMPLE_STAGES + APPROACH_STAGES)
    parser.add_argument("--approach", choices=APPROACHES)
    args = parser.parse_args(argv)

    if args.stage in APPROACH_STAGES:
        if not args.approach:
            parser.error(f"--approach requis pour l'etape {args.stage}")
        from docguard_ai.models import evaluate, export_onnx, train
        {"train": train, "evaluate": evaluate, "export": export_onnx}[args.stage].run(args.approach)
    elif args.stage == "validate":
        _validate()
    elif args.stage == "prepare":
        _prepare()
    else:
        from docguard_ai.models import baselines
        baselines.run()


if __name__ == "__main__":
    main()
