#!/usr/bin/env python3
"""Export Task 1 and evaluate ONNX on the official validation manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "research/checkpoints/task1_universal_ae_final_best.pt"
DEFAULT_CONFIG = REPO_ROOT / "research/configs/task1_best_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "backend/models/task1_universal_ae.onnx"
DEFAULT_MANIFEST = REPO_ROOT / "research/val_manifest_official.json"
DEFAULT_IMAGE_DIR = REPO_ROOT / "research/datasets/oxford-iiit-pet/images"
DEFAULT_REPORT = REPO_ROOT / "export/reports/task1_universal_ae_validation.json"
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4
IMAGE_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-validation", action="store_true", help="Only export and run ONNX/PyTorch parity on one validation image")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def fail(message: str) -> "NoReturn":
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = args.validation_manifest.expanduser().resolve()
    image_dir = args.validation_image_dir.expanduser().resolve()

    for label, path in (("Task 1 config", DEFAULT_CONFIG), ("checkpoint", checkpoint_path),
                        ("validation manifest", manifest_path), ("validation image directory", image_dir)):
        if not path.exists():
            fail(f"{label} not found: {path}")
    if args.batch_size < 1:
        fail("--batch-size must be positive")

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        import yaml
        from PIL import Image
    except ImportError as exc:
        fail(f"Missing export dependency ({exc.name}); install numpy, torch, pyyaml, onnx, onnxruntime, and pillow.")

    sys.path.insert(0, str(REPO_ROOT / "research"))
    try:
        from src.corruptions import apply_corruption
        from src.losses import compute_ssim
        from src.models_dae import ConvDAE
    except ImportError as exc:
        fail(f"Could not import Task 1 research code: {exc}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not read validation manifest {manifest_path}: {exc}")
    if not isinstance(manifest, dict) or not manifest:
        fail(f"Validation manifest is empty or invalid: {manifest_path}")

    def prepare_sample(filename: str, entry: dict):
        sample_path = (image_dir / filename).resolve()
        if sample_path.parent != image_dir or not sample_path.is_file():
            fail(f"Validation image is missing or outside the image directory: {filename}")
        with Image.open(sample_path) as source:
            clean_image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            clean = torch.from_numpy(np.asarray(clean_image, dtype=np.float32).copy() / 255.0).permute(2, 0, 1)
        kind = str(entry.get("corruption_type", "")).lower()
        seed = int(entry.get("seed", 0))
        corrupted = apply_corruption(
            clean,
            kind,
            severity_params=entry,
            rng=random.Random(seed),
        )
        return clean, corrupted, ("salt_pepper" if kind == "salt" else kind)

    try:
        import yaml
        config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        fail(f"Could not read Task 1 config {DEFAULT_CONFIG}: {exc}")
    params = config.get("params", {})
    required = ("base_channels", "latent_dim", "dropout")
    missing = [key for key in required if key not in params]
    if missing:
        fail(f"Task 1 config is missing model parameters: {', '.join(missing)}")

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        fail(f"Could not load checkpoint {checkpoint_path}: {exc}")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        fail(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")

    model = ConvDAE(
        in_channels=3,
        base_channels=int(params["base_channels"]),
        latent_dim=int(params["latent_dim"]),
        dropout=float(params["dropout"]),
        use_skip=False,
    ).eval()
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        fail(f"Checkpoint weights do not match the Task 1 config/model: {exc}")

    first_name = sorted(manifest)[0]
    try:
        _, first_corrupted, _ = prepare_sample(first_name, manifest[first_name])
    except Exception as exc:
        fail(f"Could not prepare first validation sample: {exc}")
    example = first_corrupted.unsqueeze(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            torch_output = model(example)
        torch.onnx.export(
            model,
            example,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except Exception as exc:
        fail(f"ONNX export failed: {exc}")

    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_output = session.run([output_name], {input_name: example.numpy()})[0]
        np.testing.assert_allclose(torch_output.cpu().numpy(), onnx_output, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    except Exception as exc:
        fail(f"Exported ONNX model failed graph, runtime, or parity validation: {exc}")

    print(f"Exported and verified: {output_path}")
    print(f"Initial validation sample: {first_name}; parity tolerances rtol={PARITY_RTOL}, atol={PARITY_ATOL}")
    if args.skip_validation:
        return

    totals = defaultdict(lambda: {"count": 0, "mse": 0.0, "psnr_db": 0.0, "ssim": 0.0})
    max_abs_error = 0.0
    rows = []
    try:
        for filename in sorted(manifest):
            clean, corrupted, kind = prepare_sample(filename, manifest[filename])
            rows.append((filename, kind, clean, corrupted))
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            clean_batch = torch.stack([row[2] for row in batch])
            corrupted_batch = torch.stack([row[3] for row in batch])
            with torch.inference_mode():
                torch_batch_output = model(corrupted_batch).clamp(0.0, 1.0)
            ort_batch_output = session.run(
                [output_name], {input_name: corrupted_batch.numpy().astype(np.float32)}
            )[0]
            np.testing.assert_allclose(
                torch_batch_output.numpy(), ort_batch_output, rtol=PARITY_RTOL, atol=PARITY_ATOL
            )
            max_abs_error = max(
                max_abs_error,
                float(np.max(np.abs(torch_batch_output.numpy() - ort_batch_output))),
            )
            ort_output = torch.from_numpy(ort_batch_output).clamp(0.0, 1.0)
            mse = (ort_output - clean_batch).square().flatten(1).mean(1)
            psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
            ssim_map = compute_ssim(ort_output, clean_batch, reduction="none")
            ssim = ssim_map.flatten(1).mean(1)
            for i, row in enumerate(batch):
                item = totals[row[1]]
                item["count"] += 1
                item["mse"] += float(mse[i])
                item["psnr_db"] += float(psnr[i])
                item["ssim"] += float(ssim[i])
    except Exception as exc:
        fail(f"Validation-set inference failed: {exc}")

    by_corruption = {
        kind: {key: value / stats["count"] for key, value in stats.items() if key != "count"}
        | {"count": stats["count"]}
        for kind, stats in sorted(totals.items())
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "onnx_model": str(output_path),
        "validation_manifest": str(manifest_path),
        "samples": len(rows),
        "max_pytorch_onnx_abs_error": max_abs_error,
        "parity_rtol": PARITY_RTOL,
        "parity_atol": PARITY_ATOL,
        "metrics_by_corruption": by_corruption,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} official validation images; max PyTorch/ONNX abs error: {max_abs_error:.6g}")
    for kind, metrics in by_corruption.items():
        print(f"{kind}: n={metrics['count']} PSNR={metrics['psnr_db']:.3f} dB SSIM={metrics['ssim']:.4f}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
