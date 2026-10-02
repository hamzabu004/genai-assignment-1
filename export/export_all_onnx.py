#!/usr/bin/env python3
"""Export and verify the seven final-best deployment models as ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "research/checkpoints"
OUTPUT_DIR = ROOT / "backend/models"
REPORT = ROOT / "export/reports/final_best_onnx_validation.json"
ATOL = 1e-4
RTOL = 1e-4
IMAGE_SHAPE = (1, 3, 128, 128)

MODEL_SPECS = [
    ("task1_universal_ae_final_best.pt", "task1_universal_ae.onnx", "task1"),
    ("task2a_classifier_final_best.pt", "task2_classifier.onnx", "classifier"),
    ("task2b_specialist_salt_final_best.pt", "task2_specialist_salt.onnx", "specialist"),
    ("task2b_specialist_blur_final_best.pt", "task2_specialist_blur.onnx", "specialist"),
    ("task2b_specialist_occlusion_final_best.pt", "task2_specialist_occlusion.onnx", "specialist"),
    ("task3_soft_moe_final_best.pt", "task3_soft_moe.onnx", "moe"),
    ("task4_generator_final_best.pt", "task4_generator.onnx", "generator"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    missing = [str(CHECKPOINTS / name) for name, _, _ in MODEL_SPECS if not (CHECKPOINTS / name).is_file()]
    if missing:
        raise SystemExit("Missing final-best checkpoints:\n  " + "\n  ".join(missing))

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        import yaml
    except ImportError as exc:
        raise SystemExit(
            f"Missing export dependency '{exc.name}'. Activate the research environment and install "
            "torch, numpy, pyyaml, onnx, onnxruntime, and pillow."
        ) from exc

    sys.path.insert(0, str(ROOT / "research"))
    from src.models_classifier import CorruptionClassifier, SoftMoERestorer
    from src.models_dae import ConvDAE
    from src.models_gan import UNetGenerator

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    device = torch.device("cpu")

    def checkpoint(name: str) -> dict:
        path = CHECKPOINTS / name
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        if not isinstance(value, dict) or not isinstance(value.get("model_state_dict"), dict):
            raise RuntimeError(f"Invalid checkpoint (missing model_state_dict): {path}")
        return value

    def make_dae(meta: dict, state: dict | None = None) -> ConvDAE:
        if state is not None:
            base = int(state["enc1.0.weight"].shape[0])
            latent = int(state["fc_z.weight"].shape[0])
        else:
            base = int(meta["base_channels"])
            latent = int(meta["latent_dim"])
        return ConvDAE(3, base, latent, float(meta.get("dropout", 0.0)), use_skip=False).eval()

    def build(kind: str, meta: dict):
        state = meta["model_state_dict"]
        if kind == "task1":
            cfg_path = ROOT / "research/configs/task1_best_config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            params = cfg.get("params", {})
            required = ("base_channels", "latent_dim", "dropout")
            absent = [key for key in required if key not in params]
            if absent:
                raise RuntimeError(f"Task 1 config lacks: {', '.join(absent)}")
            model = ConvDAE(3, int(params["base_channels"]), int(params["latent_dim"]),
                            float(params["dropout"]), use_skip=False).eval()
            model.load_state_dict(state, strict=True)
            return model, (torch.from_numpy(rng.random(IMAGE_SHAPE, dtype=np.float32)),), ["input"], ["output"], [{0: "batch"}], [{0: "batch"}]
        if kind == "classifier":
            model = CorruptionClassifier(
                base_channels=int(meta["base_channels"]), dropout=float(meta.get("dropout", 0.0))
            ).eval()
            model.load_state_dict(state, strict=True)
            return model, (torch.from_numpy(rng.random(IMAGE_SHAPE, dtype=np.float32)),), ["input"], ["logits"], [{0: "batch"}], [{0: "batch"}]
        if kind == "specialist":
            model = make_dae(meta)
            model.load_state_dict(state, strict=True)
            return model, (torch.from_numpy(rng.random(IMAGE_SHAPE, dtype=np.float32)),), ["input"], ["output"], [{0: "batch"}], [{0: "batch"}]
        if kind == "moe":
            gate_state = {key.removeprefix("gate."): val for key, val in state.items() if key.startswith("gate.")}
            gate = CorruptionClassifier(
                base_channels=int(gate_state["conv1.0.weight"].shape[0]), dropout=0.0
            ).eval()
            gate.load_state_dict(gate_state, strict=True)
            experts = []
            for name in ("specialist_salt", "specialist_blur", "specialist_occlusion"):
                prefix = name + "."
                expert_state = {key[len(prefix):]: val for key, val in state.items() if key.startswith(prefix)}
                expert = make_dae({}, expert_state).eval()
                expert.load_state_dict(expert_state, strict=True)
                experts.append(expert)
            model = SoftMoERestorer(gate, *experts, temperature=float(meta.get("temperature", 1.0))).eval()
            model.load_state_dict(state, strict=True)
            return model, (torch.from_numpy(rng.random(IMAGE_SHAPE, dtype=np.float32)),), ["input"], \
                ["output", "routing_probs", "gate_logits"], [{0: "batch"}], [{0: "batch"}, {0: "batch"}, {0: "batch"}]
        if kind == "generator":
            arch = meta.get("architecture", {})
            model = UNetGenerator(
                base_channels=int(meta["base_channels"]), embed_dim=int(meta["embed_dim"]),
                dropout=float(meta.get("dropout", 0.0)),
                multi_scale_style=bool(arch.get("multi_scale_style", True)),
                grayscale_constraint=bool(arch.get("grayscale_constraint", True)),
                residual_refinement=bool(meta.get("residual_refinement", arch.get("residual_refinement", False))),
            ).eval()
            model.load_state_dict(state, strict=True)
            photo = torch.from_numpy(rng.random(IMAGE_SHAPE, dtype=np.float32) * 2.0 - 1.0)
            style = torch.tensor([0], dtype=torch.int64)
            return model, (photo, style), ["photo", "style_idx"], ["output"], \
                [{0: "batch"}, {0: "batch"}], [{0: "batch"}]
        raise ValueError(kind)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_items = []
    for ckpt_name, onnx_name, kind in MODEL_SPECS:
        ckpt = checkpoint(ckpt_name)
        model, examples, input_names, output_names, input_axes, output_axes = build(kind, ckpt)
        output_path = output_dir / onnx_name
        try:
            with torch.inference_mode():
                torch_outputs = model(*examples)
                if not isinstance(torch_outputs, (tuple, list)):
                    torch_outputs = (torch_outputs,)
                torch.onnx.export(
                    model, examples, str(output_path), input_names=input_names,
                    output_names=output_names, dynamic_axes=dict(zip(input_names, input_axes)) | dict(zip(output_names, output_axes)),
                    opset_version=17, do_constant_folding=True, dynamo=False,
                )
            graph = onnx.load(str(output_path))
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            feed = {arg.name: example.detach().cpu().numpy() for arg, example in zip(session.get_inputs(), examples)}
            ort_outputs = session.run(None, feed)
            if len(ort_outputs) != len(torch_outputs):
                raise RuntimeError(f"output count mismatch: PyTorch {len(torch_outputs)}, ONNX {len(ort_outputs)}")
            errors = []
            style_errors = []
            for reference, actual in zip(torch_outputs, ort_outputs):
                np.testing.assert_allclose(reference.detach().cpu().numpy(), actual, rtol=RTOL, atol=ATOL)
                errors.append(float(np.max(np.abs(reference.detach().cpu().numpy() - actual))))
            if kind == "generator":
                style_input = session.get_inputs()[1].name
                photo_input = session.get_inputs()[0].name
                for style_id in range(3):
                    style_value = np.asarray([style_id], dtype=np.int64)
                    actual = session.run(None, {photo_input: examples[0].numpy(), style_input: style_value})[0]
                    with torch.inference_mode():
                        expected = model(examples[0], torch.from_numpy(style_value)).numpy()
                    np.testing.assert_allclose(expected, actual, rtol=RTOL, atol=ATOL)
                    style_errors.append(float(np.max(np.abs(expected - actual))))
            digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
            report_items.append({
                "checkpoint": ckpt_name, "onnx_model": onnx_name,
                "bytes": output_path.stat().st_size, "sha256": digest,
                "max_abs_error": max(errors, default=0.0), "rtol": RTOL, "atol": ATOL,
                "inputs": input_names, "outputs": output_names,
            })
            if kind == "generator":
                report_items[-1]["max_abs_error_by_style_id"] = style_errors
            print(f"Exported and verified {output_path} (max abs error {max(errors, default=0.0):.3g})")
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed exporting {ckpt_name} -> {onnx_name}: {exc}") from exc

    report = {"models": report_items, "count": len(report_items), "opset": 17}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Exported and verified {len(report_items)} models. Report: {report_path}")


if __name__ == "__main__":
    main()
