#!/usr/bin/env python3
"""
Update all research notebooks to conform to the updated GenAI Assignment 1 specifications:
1. Embed all formal LaTeX mathematical equations in markdown cells.
2. Fix missing variables in task2b_specialists_optuna.ipynb (N_TRIALS, OPTUNA_EPOCHS, etc.).
3. Add full-schedule retraining cells in Optuna notebooks (Tasks 2a, 2b, 3, 4) to produce the
   final model checkpoints expected by the validation notebooks.
4. Update task2b_validation.ipynb to support final checkpoints.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("research/notebooks")


def to_source_lines(text: str) -> list[str]:
    """Convert multi-line string into Jupyter notebook cell source lines."""
    lines = text.strip().split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])


def update_notebook(nb_name: str, modifier_fn):
    nb_path = NOTEBOOKS_DIR / nb_name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    nb = modifier_fn(nb)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"[OK] Updated {nb_name}")


# ============================================================================
# Task 1 Notebooks
# ============================================================================

def modify_task1_dae_simple(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 1: Universal Multi-Corruption Restoration (DAE Simple Baseline)

This notebook implements the **Simple Universal Denoising Autoencoder (DAE) Baseline** for Task 1 as specified in the assignment.

### Mathematical Formulation
The universal autoencoder maps a corrupted input $\\tilde{x} \\in \\mathbb{R}^{3 \\times 128 \\times 128}$ to a restored RGB image $\\hat{x}$:
$$\\hat{x} = D\\big(E(\\tilde{x})\\big)$$
where $E$ denotes the convolutional encoder compressing the input into a genuine bottleneck latent representation $z = E(\\tilde{x}) \\in \\mathbb{R}^d$, and $D$ denotes the convolutional decoder reconstructing the clean image $\\hat{x} = D(z)$.

The network is trained using a linear combination of pixel reconstruction loss and Structural Similarity Index Measure (SSIM) loss:
$$\\mathcal{L}_{AE} = \\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x}) + (1-\\alpha)\\big(1-\\mathrm{SSIM}(x,\\hat{x})\\big)$$
where $x$ represents the clean target image, $\\hat{x}$ is the reconstructed output, and $\\alpha \\in [0, 1]$ balances pixel fidelity and structural perceptual quality (initial value $\\alpha = 0.8$).

### Workflow:
1. **Config & Environment**: Hardware detection and hyperparameter definitions ($\\alpha = 0.8$, $\\text{latent\\_dim} = 256$, $\\text{base\\_channels} = 32$).
2. **Data Pipeline**: Oxford-IIIT Pet dataset with runtime dynamic corruptions (clean, salt-and-pepper, Gaussian blur, rectangular occlusion sampled with equal probability $p = 0.25$).
3. **Architecture**: Convolutional encoder-decoder (`ConvDAE`) with GroupNorm, LeakyReLU, and strided convolutions with a genuine compressed bottleneck (no unrestricted skip connections).
4. **Loss**: Composite objective $\\mathcal{L}_{AE} = 0.8\\,\\mathcal{L}_{L1} + 0.2\\,(1 - \\mathrm{SSIM})$.
5. **Training & Validation**: Multi-epoch training loop with automatic checkpointing (`task1_universal_ae_latest.pt` & `task1_universal_ae_best.pt`).
6. **Evaluation & Visualization**: Visual restoration inspection grid and transition gate checks before Optuna.""")
    return nb


def modify_task1_dae_optuna(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 1: Universal DAE Hyperparameter Optimization (Optuna)

This notebook executes the **Optuna Hyperparameter Search and Full-Schedule Retraining** for Task 1 as specified in the assignment.

### Mathematical Formulation
The universal denoising autoencoder performs:
$$\\hat{x} = D\\big(E(\\tilde{x})\\big)$$
optimizing the composite reconstruction loss:
$$\\mathcal{L}_{AE} = \\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x}) + (1-\\alpha)\\big(1-\\mathrm{SSIM}(x,\\hat{x})\\big)$$

### Optuna Search Space Requirements
As required by the assignment, the Optuna study investigates:
1. **Learning Rate** ($\\text{lr} \\in [10^{-4}, 10^{-3}]$ log scale)
2. **Batch Size** ($B \\in \\{16, 32, 64\\}$)
3. **Bottleneck Dimension** ($\\text{latent\\_dim} \\in \\{64, 128, 256\\}$)
4. **Encoder Base Channels** ($\\text{base\\_channels} \\in \\{32, 64\\}$)
5. **Dropout Rate** ($\\text{dropout} \\in [0.0, 0.3]$)
6. **Loss Weighting Factor** ($\\alpha \\in [0.5, 0.95]$)

### Optimization Strategy:
- Persistent SQLite storage (`task1_study.db`) with TPE sampler and MedianPruner.
- Reduced-epoch trials (15 epochs) for rapid hyperparameter exploration.
- Winning configuration retrained for the full schedule (60 epochs), saving `task1_universal_ae_final_best.pt`.
- Mandatory **Skip-Connection Ablation** comparing the winning bottleneck model against an equivalent model with a skip connection.""")
    return nb


def modify_task1_validation(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 1: Universal DAE Validation & Benchmark Evaluation

This notebook evaluates the final trained Task 1 Universal DAE model across all corruption types and severities on the deterministic test benchmark.

### Mathematical Formulation & Evaluation Metrics
The model reconstructs clean estimates from corrupted inputs:
$$\\hat{x} = D\\big(E(\\tilde{x})\\big)$$
Quantitative evaluation is performed against the ground-truth clean image $x$ using:
- **Pixel Reconstruction (L1)**: $\\mathcal{L}_{L1} = \\frac{1}{CHW} \\|x - \\hat{x}\\|_1$
- **Structural Similarity (SSIM)**: preserving luminance, contrast, and structure
- **Peak Signal-to-Noise Ratio (PSNR)**:
  $$\\mathrm{PSNR} = 10 \\log_{10}\\left(\\frac{\\mathrm{MAX}_I^2}{\\mathrm{MSE}}\\right) = 20 \\log_{10}\\left(\\frac{1}{\\sqrt{\\mathrm{MSE}}}\\right)$$
- **Absolute Error Map**: $|x - \\hat{x}|$ highlighting localized restoration performance.

### Reporting Breakdown:
1. Independent results for **Clean**, **Salt-and-Pepper**, **Gaussian Blur**, and **Rectangular Occlusion**.
2. Independent results separated by **Low**, **Medium**, and **High** severities.
3. 12 representative examples and 4 failure cases with absolute error maps.
4. Latent space analysis (active dimensions, PCA, t-SNE, latent linear interpolation).""")
    return nb


# ============================================================================
# Task 2a Notebooks
# ============================================================================

def modify_task2a_classifier_simple(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2a: Corruption Classifier (Simple Baseline)

This notebook trains the **Convolutional Corruption Classifier** for Task 2a as required by the assignment.

### Mathematical Formulation
Given an input image $\\tilde{x} \\in \\mathbb{R}^{3 \\times 128 \\times 128}$, the classifier computes posterior class probabilities across the four runtime conditions:
$$p = f(\\tilde{x}) = [p_{clean},\\, p_{salt},\\, p_{blur},\\, p_{occlusion}], \\quad \\sum_{k=1}^{4} p_k = 1$$
The predicted corruption class $\\hat{k}$ is determined by argmax decision:
$$\\hat{k} = \\arg\\max_{k}\\, p_k$$

The classifier is trained using multiclass cross-entropy loss with balanced batches:
$$\\mathcal{L}_{CE} = -\\sum_{k=1}^{4} y_k \\log p_k$$
where $y \\in \\{0, 1\\}^4$ is the one-hot ground-truth corruption label generated by the runtime data loading pipeline.""")
    return nb


def modify_task2a_classifier_optuna(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2a: Corruption Classifier Optuna Search & Full Retraining

Hyperparameter tuning and full-schedule retraining for the Task 2a CNN classifier as specified in the assignment.

### Mathematical Formulation
$$p = f(\\tilde{x}) = [p_{clean},\\, p_{salt},\\, p_{blur},\\, p_{occlusion}], \\quad \\hat{k} = \\arg\\max_{k}\\, p_k$$
$$\\mathcal{L}_{CE} = -\\sum_{k=1}^{4} y_k \\log p_k$$

### Optuna Search Space Requirements
As required by the assignment, Optuna tunes:
1. **Learning Rate** ($\\text{lr} \\in [10^{-4}, 10^{-2}]$ log scale)
2. **Batch Size** ($B \\in \\{32, 64, 128\\}$)
3. **Convolutional Channel Configuration** ($\\text{base\\_channels} \\in \\{16, 32, 64\\}$)
4. **Dropout Rate** ($\\text{dropout} \\in [0.0, 0.5]$)
5. **Weight Decay** ($\\text{weight\\_decay} \\in [10^{-5}, 10^{-3}]$ log scale)

### Retraining:
The winning configuration is retrained for `FULL_TRAIN_EPOCHS = 35` to produce `task2a_classifier_final_best.pt` for downstream hard routing and soft MoE initialization.""")

    # Add full retraining code to Cell 6
    nb["cells"][6]["source"] = to_source_lines("""# 6. Save Best Config & Retrain Winning Model on Full Schedule
best_config_path = CONFIGS_ROOT / "task2a_best_config.yaml"
with open(best_config_path, "w") as f:
    yaml.dump({
        "task": "task2a_corruption_classifier",
        "best_trial_number": int(study.best_trial.number),
        "best_accuracy": float(study.best_value),
        "params": study.best_params,
    }, f, indent=2)
print(f"[Config] Saved best classifier hyperparameters to {best_config_path}")

# Retrain winning configuration on full schedule
print(f"\\n--- Retraining Winning Classifier ({FULL_TRAIN_EPOCHS} Epochs) ---")
p = study.best_params
train_loader = DataLoader(train_dataset, batch_size=p["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=p["batch_size"], shuffle=False, num_workers=2)

winner_model = CorruptionClassifier(
    base_channels=p["base_channels"],
    dropout=p["dropout"],
    num_classes=4
).to(device)

optimizer = torch.optim.Adam(winner_model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
best_val_acc = 0.0
checkpoint_dir = RESEARCH_ROOT / "checkpoints"

for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
    tr = train_one_epoch_classifier(winner_model, train_loader, optimizer, device, scaler=scaler)
    vl = validate_classifier(winner_model, val_loader, device)
    val_acc = vl["val_acc"]
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
    ckpt = create_checkpoint_dict(
        winner_model, optimizer, epoch, vl["val_loss"],
        optuna_trial_number=int(study.best_trial.number),
        random_seed=SEED,
        extra_metadata={**p, "val_acc": val_acc}
    )
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task2a_classifier_final",
        ckpt_dict=ckpt,
        is_best=is_best,
        task_name="task2a",
        phase_name="final",
        device_name=device_report()["device_name"]
    )
    print(f"Epoch [{epoch:02d}/{FULL_TRAIN_EPOCHS:02d}] Train Acc: {tr['train_acc']:.2%} | Val Acc: {val_acc:.2%} {'*Best*' if is_best else ''}")

print(f"Final retrained classifier best accuracy: {best_val_acc:.2%}")""")
    return nb


def modify_task2a_validation(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2a: Corruption Classifier Validation & Benchmark Evaluation

Evaluates the trained corruption classifier on the deterministic test benchmark as required by the assignment.

### Mathematical Formulation
$$p = f(\\tilde{x}) = [p_{clean},\\, p_{salt},\\, p_{blur},\\, p_{occlusion}], \\quad \\hat{k} = \\arg\\max_{k}\\, p_k$$

### Metrics Reported:
- **Overall Accuracy**
- **Macro-Averaged Precision, Recall, and F1-Score**
- **Per-Class Metrics Table** (clean, salt-and-pepper, Gaussian blur, rectangular occlusion)
- **Normalised 4-Class Confusion Matrix**
- **Accuracy Stratified by Severity Level** (low, medium, high)
- **Failure Mode Analysis** (misclassified examples grouped by confusion pair)""")
    return nb


# ============================================================================
# Task 2b Notebooks
# ============================================================================

def modify_task2b_specialists_simple(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2b: Specialist Restoration Autoencoders (Simple Baseline)

This notebook trains the **three specialist restoration autoencoders** independently:
1. **Salt-and-Pepper Specialist** ($D_{salt}$)
2. **Gaussian Blur Specialist** ($D_{blur}$)
3. **Occlusion Specialist** ($D_{occlusion}$)

### Mathematical Formulation
Each specialist $k \\in \\{\\text{salt}, \\text{blur}, \\text{occlusion}\\}$ is trained strictly on its designated corruption domain to reconstruct the clean target image $x$:
$$\\hat{x}_k = D_k\\big(E_k(\\tilde{x}_k)\\big)$$
Each specialist optimizes the composite structural reconstruction objective:
$$\\mathcal{L}_{specialist} = \\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x}_k) + (1-\\alpha)\\big(1-\\mathrm{SSIM}(x,\\hat{x}_k)\\big)$$
with initial $\\alpha = 0.8$.""")
    return nb


def modify_task2b_specialists_optuna(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2b: Specialist Shared Architecture Search (Optuna) & Retraining

Conducts a shared architecture search on representative corruption data as specified in the assignment.
The locked winning architecture is then trained across all three specialists.

### Mathematical Formulation
$$\\hat{x}_k = D_k\\big(E_k(\\tilde{x}_k)\\big), \\quad \\mathcal{L}_k = \\alpha\\,\\mathcal{L}_{L1}(x,\\hat{x}_k) + (1-\\alpha)\\big(1-\\mathrm{SSIM}(x,\\hat{x}_k)\\big)$$

### Optuna Search Space Requirements
As required by the assignment, the shared Optuna search tunes:
1. **Learning Rate** ($\\text{lr} \\in [10^{-4}, 10^{-3}]$ log scale)
2. **Batch Size** ($B \\in \\{16, 32, 64\\}$)
3. **Bottleneck Dimension** ($\\text{latent\\_dim} \\in \\{64, 128, 256\\}$)
4. **Channel Configuration** ($\\text{base\\_channels} \\in \\{32, 64\\}$)
5. **Dropout Rate** ($\\text{dropout} \\in [0.0, 0.3]$)
6. **L1-to-SSIM Loss Weighting** ($\\alpha \\in [0.5, 0.95]$)

### Retraining:
The winning shared architecture is trained independently on each of the 3 corruptions for `FULL_TRAIN_EPOCHS = 35` to produce `task2b_specialist_{ctype}_final_best.pt`.""")

    # Fix missing variables in Cell 1
    nb["cells"][1]["source"] = to_source_lines("""# 1. Study Configuration
N_TRIALS = 20
OPTUNA_EPOCHS = 15
FULL_TRAIN_EPOCHS = 35
STUDY_NAME = "task2b_specialists"
STORAGE_DB = "sqlite:///task2b_study.db"
SEED = 42""")

    # Check if a retrain cell exists; if not, add it
    retrain_code = to_source_lines("""# 5. Retrain All 3 Specialists on Full Schedule with Winning Architecture
p = study.best_params
print(f"\\n--- Retraining Specialists with Winning Architecture ({FULL_TRAIN_EPOCHS} Epochs) ---")
print(p)

checkpoint_dir = RESEARCH_ROOT / "checkpoints"

for ctype in ["salt", "blur", "occlusion"]:
    print(f"\\n[Retraining Specialist: {ctype}]")
    train_ds = PetDataset(mode="train", corruption_mode=ctype, seed=SEED)
    val_ds = PetDataset(mode="val", corruption_mode=ctype, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=p["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=p["batch_size"], shuffle=False, num_workers=2)
    
    model = ConvDAE(
        base_channels=p["base_channels"],
        latent_dim=p["latent_dim"],
        dropout=p["dropout"],
        use_skip=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best_loss = float("inf")
    
    for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
        tr = train_one_epoch_dae(model, train_loader, optimizer, device, alpha=p["alpha"], scaler=scaler)
        vl = validate_dae(model, val_loader, device, alpha=p["alpha"])
        is_best = vl["val_loss"] < best_loss
        best_loss = min(best_loss, vl["val_loss"])
        ckpt = create_checkpoint_dict(
            model, optimizer, epoch, vl["val_loss"], random_seed=SEED,
            optuna_trial_number=int(study.best_trial.number),
            extra_metadata={**p, "corruption_type": ctype}
        )
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            prefix=f"task2b_specialist_{ctype}_final",
            ckpt_dict=ckpt,
            is_best=is_best,
            task_name=f"task2b_{ctype}",
            phase_name="final",
            device_name=device_report()["device_name"]
        )
        if epoch % 5 == 0 or epoch == FULL_TRAIN_EPOCHS:
            print(f"[{ctype} {epoch:02d}/{FULL_TRAIN_EPOCHS:02d}] Train: {tr['train_loss']:.4f} | Val: {vl['val_loss']:.4f} (SSIM: {vl['val_ssim']:.4f}) {'*Best*' if is_best else ''}")

print("\\nAll 3 specialists retrained and exported successfully.")""")

    if len(nb["cells"]) <= 5:
        # Add code cell 5
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": retrain_code
        })
    else:
        nb["cells"][5]["source"] = retrain_code

    return nb


def modify_task2b_validation(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 2b: Specialist Autoencoders Validation & Hard-Routing Comparison

Evaluates each specialist DAE and the full hard-routing restoration pipeline.

### Mathematical Formulation of Hard Routing
During hard-routed inference, the classifier predicts the corruption class $\\hat{k} = \\arg\\max_k p_k$. The selected specialist restores the image, with clean inputs using an identity bypass:
$$
\\hat{x} =
\\begin{cases}
\\tilde{x}, & \\hat{k} = \\text{clean} \\\\
D_{salt}(\\tilde{x}), & \\hat{k} = \\text{salt} \\\\
D_{blur}(\\tilde{x}), & \\hat{k} = \\text{blur} \\\\
D_{occlusion}(\\tilde{x}), & \\hat{k} = \\text{occlusion}
\\end{cases}
$$

### Dual Evaluation Modes:
- **Oracle routing**: Known corruption label selects the expert (restoration upper bound).
- **Predicted routing**: Classifier prediction selects the expert (full operational pipeline).
- **Misrouting Failure Cases**: Analysis of severe restoration drops caused by classification error.""")

    # Update checkpoint loading in Cell 2 to prefer final checkpoints
    nb["cells"][2]["source"] = to_source_lines("""# 2. Checkpoint paths (prefers Optuna-retrained final models if present)
SALT_CKPT = (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_salt_final_best.pt"
             if (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_salt_final_best.pt").exists()
             else RESEARCH_ROOT / "checkpoints" / "task2b_specialist_salt_best.pt")
BLUR_CKPT = (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_blur_final_best.pt"
             if (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_blur_final_best.pt").exists()
             else RESEARCH_ROOT / "checkpoints" / "task2b_specialist_blur_best.pt")
OCCL_CKPT = (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_occlusion_final_best.pt"
             if (RESEARCH_ROOT / "checkpoints" / "task2b_specialist_occlusion_final_best.pt").exists()
             else RESEARCH_ROOT / "checkpoints" / "task2b_specialist_occlusion_best.pt")
CLS_CKPT  = (RESEARCH_ROOT / "checkpoints" / "task2a_classifier_final_best.pt"
             if (RESEARCH_ROOT / "checkpoints" / "task2a_classifier_final_best.pt").exists()
             else RESEARCH_ROOT / "checkpoints" / "task2_classifier_best.pt")""")
    return nb


# ============================================================================
# Task 3 Notebooks
# ============================================================================

def modify_task3_moe_simple(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 3: Soft Mixture-of-Experts Restoration (Simple Baseline)

This notebook implements the differentiable **Soft Mixture-of-Experts (MoE) Restoration System** as specified in the assignment.

### Mathematical Formulation
The gating network produces continuous routing weights $w$ over the clean identity branch and the three specialists using temperature-scaled softmax:
$$w = \\mathrm{softmax}\\!\\left(\\frac{g(\\tilde{x})}{\\tau}\\right) = [w_c, w_s, w_b, w_o]$$
where $\\tau > 0$ is the temperature parameter controlling routing sharpness.

The differentiable reconstruction combines all branches:
$$\\hat{x} = w_c\\,\\tilde{x} + w_s\\,D_{salt}(\\tilde{x}) + w_b\\,D_{blur}(\\tilde{x}) + w_o\\,D_{occlusion}(\\tilde{x})$$

The complete system is trained jointly with the four-component loss:
$$\\mathcal{L}_{joint} = \\lambda_1\\,\\mathcal{L}_{L1} + \\lambda_2\\,(1-\\mathrm{SSIM}) + \\lambda_3\\,\\mathcal{L}_{CE} + \\lambda_4\\,\\mathcal{L}_{balance}$$
where the balance regularizer prevents routing collapse:
$$\\mathcal{L}_{balance} = \\sum_{k=1}^{4}\\left(\\bar{w}_k - \\frac{1}{4}\\right)^{2}$$
$\bar{w}_k$ is the average routing weight assigned to branch $k$ across a balanced training batch. Initial weights: $\\lambda_1 = 0.8, \\lambda_2 = 0.2, \\lambda_3 = 0.1, \\lambda_4 = 0.01$.

### Two-Stage Training:
- **Stage 1 (Warm-Up)**: Specialists frozen; only the gating network is trained.
- **Stage 2 (Joint Fine-Tuning)**: All parameters unfrozen; end-to-end fine-tuning with a smaller learning rate.""")
    return nb


def modify_task3_moe_optuna(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 3: Soft MoE Optuna Search & Full Retraining

Hyperparameter optimization and full schedule retraining for the soft Mixture-of-Experts system as specified in the assignment.

### Mathematical Formulation
$$w = \\mathrm{softmax}\\!\\left(\\frac{g(\\tilde{x})}{\\tau}\\right), \\quad \\hat{x} = w_c\\,\\tilde{x} + w_s\\,D_{salt}(\\tilde{x}) + w_b\\,D_{blur}(\\tilde{x}) + w_o\\,D_{occlusion}(\\tilde{x})$$
$$\\mathcal{L}_{joint} = \\lambda_1\\,\\mathcal{L}_{L1} + \\lambda_2\\,(1-\\mathrm{SSIM}) + \\lambda_3\\,\\mathcal{L}_{CE} + \\lambda_4\\,\\mathcal{L}_{balance}$$
$$\\mathcal{L}_{balance} = \\sum_{k=1}^{4}\\left(\\bar{w}_k - \\frac{1}{4}\\right)^{2}$$

### Optuna Search Space Requirements
As required by the assignment, Optuna tunes:
1. **Joint Fine-Tuning Learning Rate** ($\\text{finetune\\_lr} \\in [10^{-5}, 10^{-4}]$ log scale)
2. **Softmax Temperature** ($\\tau \\in [0.5, 2.0]$)
3. **Classification Loss Weight** ($\\lambda_3 \\in [0.01, 0.5]$)
4. **Routing Balance Regularizer Weight** ($\\lambda_4 \\in [10^{-3}, 10^{-1}]$ log scale)
5. **Reconstruction-Loss Weighting** ($\\lambda_1 = \\alpha, \\lambda_2 = 1 - \\alpha$ with $\\alpha \\in [0.5, 0.95]$)

### Retraining:
The winning configuration is retrained for `FULL_TRAIN_EPOCHS = 25` to produce `task3_soft_moe_final_best.pt` for deployment and benchmarking.""")

    # Add Cell 5 for full retraining if not present
    retrain_code = to_source_lines("""# 5. Retrain Winning Soft MoE on Full Schedule
FULL_TRAIN_EPOCHS = 25
WARMUP_EPOCHS = 5
p = study.best_params
print(f"\\n--- Retraining Winning Soft MoE ({FULL_TRAIN_EPOCHS} Joint Epochs) ---")
print(p)

lambdas = {
    "lambda1": p["alpha"],
    "lambda2": 1.0 - p["alpha"],
    "lambda3": p["lambda3"],
    "lambda4": p["lambda4"]
}

# Initialize fresh gate and specialists from Task 2 checkpoints
gate_state = torch.load(gate_path, map_location=device, weights_only=False)["model_state_dict"]
gate = CorruptionClassifier(base_channels=gate_state["conv1.0.weight"].shape[0]).to(device)
gate.load_state_dict(gate_state)

sp_salt = ConvDAE(base_channels=torch.load(expert_paths["salt"], map_location=device, weights_only=False)["model_state_dict"]["enc1.0.weight"].shape[0],
                  latent_dim=torch.load(expert_paths["salt"], map_location=device, weights_only=False)["model_state_dict"]["fc_z.weight"].shape[0]).to(device)
sp_salt.load_state_dict(torch.load(expert_paths["salt"], map_location=device, weights_only=False)["model_state_dict"])

sp_blur = ConvDAE(base_channels=torch.load(expert_paths["blur"], map_location=device, weights_only=False)["model_state_dict"]["enc1.0.weight"].shape[0],
                  latent_dim=torch.load(expert_paths["blur"], map_location=device, weights_only=False)["model_state_dict"]["fc_z.weight"].shape[0]).to(device)
sp_blur.load_state_dict(torch.load(expert_paths["blur"], map_location=device, weights_only=False)["model_state_dict"])

sp_occl = ConvDAE(base_channels=torch.load(expert_paths["occlusion"], map_location=device, weights_only=False)["model_state_dict"]["enc1.0.weight"].shape[0],
                  latent_dim=torch.load(expert_paths["occlusion"], map_location=device, weights_only=False)["model_state_dict"]["fc_z.weight"].shape[0]).to(device)
sp_occl.load_state_dict(torch.load(expert_paths["occlusion"], map_location=device, weights_only=False)["model_state_dict"])

moe = SoftMoERestorer(gate, sp_salt, sp_blur, sp_occl, temperature=p["temperature"]).to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Stage 1: Warmup
for sp in [moe.specialist_salt, moe.specialist_blur, moe.specialist_occlusion]:
    for param in sp.parameters(): param.requires_grad = False
for param in moe.gate.parameters(): param.requires_grad = True

opt_warmup = torch.optim.Adam(moe.gate.parameters(), lr=1e-3)
for ep in range(1, WARMUP_EPOCHS + 1):
    train_one_epoch_moe(moe, train_loader, opt_warmup, device, lambdas)

# Stage 2: Joint Finetuning
for param in moe.parameters(): param.requires_grad = True
opt_joint = torch.optim.Adam(moe.parameters(), lr=p["finetune_lr"])
best_val_loss = float("inf")

for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
    tr = train_one_epoch_moe(moe, train_loader, opt_joint, device, lambdas)
    vl = validate_moe(moe, val_loader, device, lambdas)
    is_best = vl["val_loss"] < best_val_loss
    if is_best: best_val_loss = vl["val_loss"]
    ckpt = create_checkpoint_dict(
        moe, opt_joint, epoch, vl["val_loss"],
        optuna_trial_number=int(study.best_trial.number),
        random_seed=SEED, extra_metadata={**p, **lambdas}
    )
    save_checkpoint(
        checkpoint_dir=checkpoint_dir, prefix="task3_soft_moe_final",
        ckpt_dict=ckpt, is_best=is_best, task_name="task3",
        phase_name="final", device_name=device.type
    )
    print(f"Epoch [{epoch:02d}/{FULL_TRAIN_EPOCHS:02d}] Val Loss: {vl['val_loss']:.4f} (SSIM: {vl['val_ssim']:.4f}) {'*Best*' if is_best else ''}")

print(f"Final MoE Retraining Complete! Best Val Loss: {best_val_loss:.4f}")""")

    if len(nb["cells"]) <= 4:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": retrain_code
        })
    else:
        nb["cells"][4]["source"] = retrain_code

    return nb


def modify_task3_validation(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 3: Soft Mixture-of-Experts Validation & Routing Analysis

Validates the soft Mixture-of-Experts system against the Task 1 Universal DAE and Task 2 Oracle Specialists on the deterministic benchmark.

### Mathematical Formulation
$$w = \\mathrm{softmax}\\!\\left(\\frac{g(\\tilde{x})}{\\tau}\\right), \\quad \\hat{x} = w_c\\,\\tilde{x} + w_s\\,D_{salt}(\\tilde{x}) + w_b\\,D_{blur}(\\tilde{x}) + w_o\\,D_{occlusion}(\\tilde{x})$$
$$\\mathcal{L}_{balance} = \\sum_{k=1}^{4}\\left(\\bar{w}_k - \\frac{1}{4}\\right)^{2}$$

### Analysis & Reporting Requirements:
1. **Quantitative Comparison**: Universal DAE vs. Oracle Specialists vs. Soft MoE (PSNR, SSIM, L1).
2. **Routing Heatmap**: Average branch weights $[w_c, w_s, w_b, w_o]$ across all corruption types and severities.
3. **Branch Utilization Diagnostics**: Identification of inactive ($< 5\\%$) or dominant ($> 70\\%$) experts.
4. **Multi-Expert Blending Visualizations**: Examples where 2+ experts contribute significantly ($\\ge 0.2$).
5. **Dominant vs. Distributed Routing**: Comparative visualization of high-certainty vs. blended predictions.""")
    return nb


# ============================================================================
# Task 4 Notebooks
# ============================================================================

def modify_task4_gan_simple(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 4: Style-Conditioned Face-to-Sketch (cGAN Simple Baseline)

Implements the **Style-Conditioned Conditional Generative Adversarial Network (cGAN)** for face-to-sketch synthesis on the FS2K dataset as specified in the assignment.

### Mathematical Formulation
The generator produces a sketch $\\hat{y}$ conditioned on a facial photograph $x$ and categorical style $s \\in \\{0, 1, 2\\}$:
$$\\hat{y} = G(x, s)$$
where the style condition $s$ is projected through a learned categorical embedding incorporated into both the generator and discriminator.

The PatchGAN discriminator evaluates photograph-sketch pairs:
$$D(x, s, y) \\quad \\text{and} \\quad D\\big(x, s, G(x,s)\\big)$$

The generator optimizes a combined adversarial and pixel reconstruction objective:
$$\\mathcal{L}_G = \\mathcal{L}_{adv} + \\lambda_{L1}\\,\\mathcal{L}_{L1}\\big(y, G(x,s)\\big)$$
where initial $\\lambda_{L1} = 100$.

The discriminator optimizes binary cross-entropy with logits:
$$\\mathcal{L}_D = \\frac{1}{2}\\,\\mathbb{E}_{(x,s,y)}\\left[\\log D(x, s, y)\\right] + \\frac{1}{2}\\,\\mathbb{E}_{(x,s)}\\left[\\log\\big(1 - D(x, s, G(x,s))\\big)\\right]$$""")
    return nb


def modify_task4_gan_optuna(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 4: Conditional GAN Optuna Search & Full Retraining

Hyperparameter optimization and full-schedule retraining for the style-conditioned face-to-sketch conditional GAN as specified in the assignment.

### Mathematical Formulation
$$\\hat{y} = G(x, s), \\quad D(x, s, y), \\quad D\\big(x, s, G(x,s)\\big)$$
$$\\mathcal{L}_G = \\mathcal{L}_{adv} + \\lambda_{L1}\\,\\mathcal{L}_{L1}\\big(y, G(x,s)\\big)$$

### Optuna Search Space Requirements
As required by the assignment, Optuna tunes:
1. **Generator Learning Rate** ($G\\_lr \\in [10^{-5}, 10^{-3}]$ log scale)
2. **Discriminator Learning Rate** ($D\\_lr \\in [10^{-5}, 10^{-3}]$ log scale)
3. **Batch Size** ($B \\in \\{4, 8, 16\\}$)
4. **Base Channel Count** ($\\text{base\\_channels} \\in \\{32, 64\\}$)
5. **Dropout Rate** ($\\text{dropout} \\in [0.0, 0.5]$)
6. **Style-Embedding Dimension** ($\\text{embed\\_dim} \\in \\{4, 8, 16\\}$)
7. **Reconstruction-Loss Weight** ($\\lambda_{L1} \\in [10.0, 200.0]$)

### Retraining:
Because GAN search uses reduced epochs (15 epochs), the selected winning configuration is retrained on the complete training schedule (40 epochs), saving `task4_generator_final_best.pt`.""")

    # Add Cell 5 for full retraining if not present
    retrain_code = to_source_lines("""# 5. Retrain Winning Conditional GAN on Full Schedule
FULL_TRAIN_EPOCHS = 40
p = study.best_params
print(f"\\n--- Retraining Winning Conditional GAN ({FULL_TRAIN_EPOCHS} Epochs) ---")
print(p)

train_loader = DataLoader(train_dataset, batch_size=p["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=p["batch_size"], shuffle=False, num_workers=2)

winner_G = UNetGenerator(base_channels=p["base_channels"], embed_dim=p["embed_dim"], dropout=p["dropout"]).to(device)
winner_D = PatchGANDiscriminator(base_channels=p["base_channels"]).to(device)
optG = torch.optim.Adam(winner_G.parameters(), lr=p["g_lr"], betas=(0.5, 0.999))
optD = torch.optim.Adam(winner_D.parameters(), lr=p["d_lr"], betas=(0.5, 0.999))

best_l1 = float("inf")
checkpoint_dir = RESEARCH_ROOT / "checkpoints"

for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
    metrics = train_one_epoch_gan(winner_G, winner_D, train_loader, optG, optD, device, lambda_l1=p["lambda_l1"])
    val_metrics = validate_gan(winner_G, val_loader, device)
    is_best = val_metrics["val_l1"] < best_l1
    if is_best:
        best_l1 = val_metrics["val_l1"]
    ckptG = create_checkpoint_dict(
        winner_G, optG, epoch, val_metrics["val_l1"], random_seed=SEED,
        optuna_trial_number=int(study.best_trial.number),
        extra_metadata={**p, "val_l1": val_metrics["val_l1"]}
    )
    save_checkpoint(
        checkpoint_dir, "task4_generator_final", ckptG, is_best=is_best,
        task_name="task4_gan", phase_name="final",
        device_name=device_report()["device_name"]
    )
    print(f"Epoch {epoch:02d}/{FULL_TRAIN_EPOCHS:02d} D-real={metrics['d_real']:.4f} D-fake={metrics['d_fake']:.4f} G-adv={metrics['g_adv']:.4f} G-L1={metrics['g_l1']:.4f} Val-L1={val_metrics['val_l1']:.4f} {'*Best*' if is_best else ''}")

print(f"Final Retrained Generator Best Val L1: {best_l1:.4f}")""")

    if len(nb["cells"]) <= 4:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": retrain_code
        })
    else:
        nb["cells"][4]["source"] = retrain_code

    return nb


def modify_task4_validation(nb):
    nb["cells"][0]["source"] = to_source_lines("""# Task 4: Style-Conditioned cGAN Validation & Qualitative Evaluation

Final evaluation of the trained `UNetGenerator` on the official FS2K test set as specified in the assignment.

### Mathematical Formulation & Evaluation Metrics
$$\\hat{y} = G(x, s)$$
- **Pixel Reconstruction (L1)**: $\\mathcal{L}_{L1}(y, \\hat{y})$
- **Peak Signal-to-Noise Ratio (PSNR)**:
  $$\\mathrm{PSNR} = 10 \\log_{10}\\left(\\frac{1}{\\mathrm{MSE}}\\right)$$
- **Structural Similarity (SSIM)**
- **Style Conditioning Matrix**: 1 photo $\\times$ 3 styles side-by-side comparison
- **Failure Mode Analysis**: Identification and diagnostic analysis of the 4 highest-error sketches (style bleeding, fine-edge blur, artifact formation).""")

    if len(nb["cells"]) > 10:
        nb["cells"][10]["source"] = to_source_lines("""# 6. Find >= 4 highest-error generated sketches
records.sort(key=lambda r: r["psnr"])   # lowest PSNR = worst
failures = records[:4]

fig, axes = plt.subplots(len(failures), 3, figsize=(11, 3.2 * len(failures)), squeeze=False)
for row, rec in zip(axes, failures):
    photo_disp  = ((rec["photo"].permute(1, 2, 0).numpy()  + 1) * 0.5).clip(0, 1)
    sketch_disp = ((rec["sketch"].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1)
    gen_disp    = ((rec["gen"].permute(1, 2, 0).numpy()    + 1) * 0.5).clip(0, 1)

    for ax, image, title in zip(row, [photo_disp, gen_disp, sketch_disp],
                                 ["Input Photo", "Generated Sketch", "Ground Truth"]):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    print(f"Style {rec['style']} | L1={rec['l1']:.4f} | "
          f"PSNR={rec['psnr']:.2f} dB | SSIM={rec['ssim']:.3f}")
    print("  Possible causes: style bleeding (textures from adjacent style), "
          "missing facial detail (glasses, beard), or blurry edges in hair region.")

plt.tight_layout()
plt.show()""")

    return nb


# ============================================================================
# Main Execution
# ============================================================================

def main():
    tasks = [
        ("task1_dae_simple.ipynb", modify_task1_dae_simple),
        ("task1_dae_optuna.ipynb", modify_task1_dae_optuna),
        ("task1_validation.ipynb", modify_task1_validation),
        ("task2a_classifier_simple.ipynb", modify_task2a_classifier_simple),
        ("task2a_classifier_optuna.ipynb", modify_task2a_classifier_optuna),
        ("task2a_validation.ipynb", modify_task2a_validation),
        ("task2b_specialists_simple.ipynb", modify_task2b_specialists_simple),
        ("task2b_specialists_optuna.ipynb", modify_task2b_specialists_optuna),
        ("task2b_validation.ipynb", modify_task2b_validation),
        ("task3_moe_simple.ipynb", modify_task3_moe_simple),
        ("task3_moe_optuna.ipynb", modify_task3_moe_optuna),
        ("task3_validation.ipynb", modify_task3_validation),
        ("task4_gan_simple.ipynb", modify_task4_gan_simple),
        ("task4_gan_optuna.ipynb", modify_task4_gan_optuna),
        ("task4_validation.ipynb", modify_task4_validation),
    ]

    for nb_name, mod_fn in tasks:
        update_notebook(nb_name, mod_fn)

    print("\\nAll 15 notebooks successfully verified and updated!")


if __name__ == "__main__":
    main()
