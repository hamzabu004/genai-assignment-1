#!/usr/bin/env python3
"""Upload the verified ONNX bundle to W&B as a versioned model Artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass
DEFAULT_MODELS = ROOT / "backend/models"
DEFAULT_REPORT = ROOT / "export/reports/final_best_onnx_validation.json"
MODEL_FILES = [
    "task1_universal_ae.onnx", "task2_classifier.onnx",
    "task2_specialist_salt.onnx", "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx", "task3_soft_moe.onnx", "task4_generator.onnx",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-name", default="final-best-onnx")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "genai-assignment"))
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    args = parser.parse_args()

    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required. Set it in the environment or .env before uploading.")
    models_dir = args.models_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    missing = [name for name in MODEL_FILES if not (models_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing ONNX models; export and verify them first:\n  " + "\n  ".join(missing))
    if not report_path.is_file():
        raise SystemExit(f"Export verification report not found: {report_path}")
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("W&B SDK missing. Install with `pip install wandb`.") from exc

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("count") != len(MODEL_FILES):
        raise SystemExit(f"Verification report does not describe {len(MODEL_FILES)} models: {report_path}")

    run = wandb.init(project=args.project, entity=args.entity, job_type="onnx-model-upload")
    try:
        artifact = wandb.Artifact(
            name=args.artifact_name,
            type="model",
            description="Verified final-best ONNX deployment models.",
            metadata={"model_count": len(MODEL_FILES), "verification_report": report_path.name},
        )
        for filename in MODEL_FILES:
            artifact.add_file(str(models_dir / filename), name=filename)
        artifact.add_file(str(report_path), name="final_best_onnx_validation.json")
        run.log_artifact(artifact, aliases=["latest", "final-best"])
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise
    print(f"Uploaded W&B Artifact {args.entity or run.entity}/{args.project}/{args.artifact_name}:latest")


if __name__ == "__main__":
    main()
