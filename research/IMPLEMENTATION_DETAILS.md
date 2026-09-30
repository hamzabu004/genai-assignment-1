# Research Implementation Reference & Details

A compact technical reference for all notebooks, shared source modules, hyperparameters, architectures, and dataset configurations across Tasks 1–4.

---

## 1. Directory Structure Overview

```
research/
├── constants.py                     # Centralized paths, image dimensions, normalization constants
├── requirements.txt                 # Pinned dependencies for research workflows
├── checkpoints_manifest.json        # Manifest tracking metadata for all checkpoints across devices
├── IMPLEMENTATION_DETAILS.md        # This technical documentation file
├── configs/                         # Optuna best-parameter YAML exports
│   ├── task1_best_config.yaml
│   ├── task2a_best_config.yaml
│   ├── task2b_salt_best_config.yaml
│   ├── task2b_blur_best_config.yaml
│   ├── task2b_occlusion_best_config.yaml
│   ├── task3_best_config.yaml
│   └── task4_best_config.yaml
├── src/                             # Reusable, shared library code
│   ├── device_utils.py              # Cross-device hardware abstraction (CUDA/ROCm/MPS/CPU)
│   ├── corruptions.py               # Pure PyTorch tensor corruption implementations
│   ├── datasets.py                  # Dataset loaders for Pet (Tasks 1–3) and FS2K (Task 4)
│   ├── losses.py                    # SSIM, VAE ELBO, MoE joint loss, and cGAN losses
│   ├── models_vae.py                # Symmetric ConvVAE with GroupNorm and optional skip
│   ├── models_classifier.py         # Lightweight 4-class CNN & SoftMoERestorer module
│   ├── models_gan.py                # UNetGenerator (with style embedding) & PatchGAN Discriminator
│   ├── checkpoint_utils.py          # Save latest/best checkpoints & manifest updater
│   └── train_loop.py                # Training and validation loops with AMP and metric tracking
└── notebooks/                       # 15 structured Jupyter notebooks
    ├── task1_vae_simple.ipynb       # Task 1 baseline experimentation
    ├── task1_vae_optuna.ipynb       # Task 1 hyperparameter search + skip ablation
    ├── task1_validation.ipynb       # Task 1 evaluation stub
    ├── task2a_classifier_simple.ipynb   # Task 2a baseline experimentation
    ├── task2a_classifier_optuna.ipynb   # Task 2a hyperparameter search
    ├── task2a_validation.ipynb          # Task 2a evaluation stub
    ├── task2b_specialists_simple.ipynb  # Task 2b 3-specialist baseline experimentation
    ├── task2b_specialists_optuna.ipynb  # Task 2b shared architecture search + retrain
    ├── task2b_validation.ipynb          # Task 2b evaluation & oracle routing comparison stub
    ├── task3_moe_simple.ipynb       # Task 3 warm-up & joint fine-tuning experimentation
    ├── task3_moe_optuna.ipynb       # Task 3 hyperparameter search (lambdas, temperature)
    ├── task3_validation.ipynb       # Task 3 evaluation & routing heatmap stub
    ├── task4_gan_simple.ipynb       # Task 4 cGAN baseline experimentation
    ├── task4_gan_optuna.ipynb       # Task 4 cGAN hyperparameter search
    └── task4_validation.ipynb       # Task 4 evaluation & style matrix stub
```

---

## 2. Shared `src/` Library Modules

| Module | Primary Classes / Functions | Purpose |
|---|---|---|
| [`constants.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/constants.py) | `PET_IMAGES_DIR`, `FS2K_ROOT`, `NORM_01_*`, `NORM_11_*`, `IMAGE_SIZE=128`, `SEED=42` | Single source of truth for file paths and normalization constants. |
| [`src/device_utils.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/device_utils.py) | `get_device()`, `device_report()` | Detects CUDA / ROCm / Apple Metal / CPU automatically so notebooks run on any device unmodified. |
| [`src/corruptions.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/corruptions.py) | `add_salt_pepper`, `add_blur`, `add_occlusion`, `apply_corruption`, `BENCHMARK_SEVERITIES` | Pure PyTorch tensor corruption implementations operating on `[0, 1]` tensors. |
| [`src/datasets.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/datasets.py) | `PetDataset`, `FS2KDataset` | Handles 80/20 train/val split manifest generation for Pets; loads photo/sketch pairs for FS2K across styles 0, 1, 2. |
| [`src/losses.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/losses.py) | `compute_ssim`, `reconstruction_loss`, `vae_loss`, `moe_joint_loss`, `generator_loss`, `discriminator_loss` | Fully differentiable SSIM, composite restoration loss, VAE ELBO, load balancing regularizer, and pix2pix GAN losses. |
| [`src/models_vae.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/models_vae.py) | `ConvVAE` | 5-stage symmetric encoder-decoder VAE with GroupNorm, reparameterization trick, and optional single skip connection. |
| [`src/models_classifier.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/models_classifier.py) | `CorruptionClassifier`, `SoftMoERestorer` | 4-class CNN classifier (GAP + MLP) and Soft MoE module integrating gate, specialists, and clean identity bypass. |
| [`src/models_gan.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/models_gan.py) | `UNetGenerator`, `PatchGANDiscriminator` | Conditional U-Net generator with learned style embedding injection and 70x70 PatchGAN discriminator. |
| [`src/checkpoint_utils.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/checkpoint_utils.py) | `create_checkpoint_dict`, `save_checkpoint`, `load_checkpoint` | Saves `latest.pt` and `best.pt`, records metadata (git hash, epoch, val loss), and maintains `checkpoints_manifest.json`. |
| [`src/train_loop.py`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/src/train_loop.py) | `train_one_epoch_*`, `validate_*`, `set_seed` | Standardized training/validation functions supporting AMP (`torch.amp.autocast`), gradient clipping, and metric logging. |

---

## 3. Notebook Inventory

### Experimentation Notebooks (Full Code)
1. **`task1_vae_simple.ipynb`**: Trains baseline Universal ConvVAE on all 4 corruption types (clean, salt, blur, occlusion at 25% each). Includes data inspection, loss curves, reconstruction grid with absolute error maps, and transition gate checklist.
2. **`task1_vae_optuna.ipynb`**: Executes 25-trial Optuna sweep over LR, batch size, latent dim, channels, dropout, $\alpha$, and $\beta$. Retrains winning model for 60 epochs and conducts the skip-connection ablation test.
3. **`task2a_classifier_simple.ipynb`**: Trains 4-class CNN baseline to identify corruption type. Evaluates with normalized confusion matrix and classification report.
4. **`task2a_classifier_optuna.ipynb`**: Tunes classifier hyperparameters to maximize validation accuracy using MedianPruner.
5. **`task2b_specialists_simple.ipynb`**: Sequentially trains the 3 specialist autoencoders (Salt, Blur, Occlusion) strictly on their designated domain.
6. **`task2b_specialists_optuna.ipynb`**: Runs a shared architecture search on representative data and retrains all 3 specialists with the winning shared structure.
7. **`task3_moe_simple.ipynb`**: Implements 2-stage Mixture-of-Experts: Stage 1 warm-up (train gate only) followed by Stage 2 joint fine-tuning of all experts and gate with load balancing.
8. **`task3_moe_optuna.ipynb`**: Tunes joint fine-tuning learning rate, softmax temperature $\tau$, classification weight $\lambda_3$, and balance weight $\lambda_4$.
9. **`task4_gan_simple.ipynb`**: Trains style-conditioned pix2pix cGAN on FS2K photo-sketch pairs. Visualizes 1 photo rendered across all 3 styles side-by-side.
10. **`task4_gan_optuna.ipynb`**: Tunes G/D learning rates, base channels, style embedding dimension, and $\lambda_{L1}$.

### Validation Stub Notebooks (Imports + Headers + TODOs)
- **`task1_validation.ipynb`**: Benchmark evaluation grid across 3 severities, error maps, and posterior collapse check.
- **`task2a_validation.ipynb`**: Confusion matrix, per-class F1, and misclassification error analysis.
- **`task2b_validation.ipynb`**: Per-specialist metrics and Oracle vs. Predicted Routing comparison.
- **`task3_validation.ipynb`**: MoE quantitative metrics vs. Task 1/2b, routing weight heatmap, and multi-expert blending cases.
- **`task4_validation.ipynb`**: Test set qualitative evaluation, style conditioning matrix, and failure mode analysis.

---

## 4. Hyperparameter Reference (Plans 6 & 7)

| Parameter | Task 1 (Universal VAE) | Task 2a (Classifier) | Task 2b (Specialists) | Task 3 (Soft MoE) | Task 4 (cGAN) |
|---|---|---|---|---|---|
| **Input Size** | 128×128×3 | 128×128×3 | 128×128×3 | 128×128×3 | 128×128×3 |
| **Pixel Normalization** | `[0, 1]` (`/255`) | `[0, 1]` (`/255`) | `[0, 1]` (`/255`) | `[0, 1]` (`/255`) | `[-1, 1]` (`/127.5 - 1`) |
| **Final Activation** | Sigmoid | Softmax (4 classes) | Sigmoid | Softmax blend | Tanh |
| **Batch Size** | 32 (range 16–64) | 64 (range 32–128) | 32 (range 16–64) | 32 (range 16–64) | 8 (range 4–16) |
| **Learning Rate** | 1e-3 (1e-4–1e-3) | 1e-3 (1e-4–1e-2) | 1e-3 (1e-4–1e-3) | Warmup 1e-3 / Fine 1e-4 | G: 2e-4, D: 2e-4 |
| **Optimizer** | Adam (0.9, 0.999) | Adam (0.9, 0.999) | Adam (0.9, 0.999) | Adam (0.9, 0.999) | Adam (0.5, 0.999) |
| **Latent / Embed Dim** | 128 (64–256) | — | 128 (64–256) | — | Style embed: 8 (4–16) |
| **Recon Loss Weights** | $\alpha=0.8$ (0.5–0.95) | CrossEntropy | $\alpha=0.8$ (0.5–0.95) | $\lambda_1=0.8, \lambda_2=0.2$ | $\lambda_{L1}=100.0$ (10–200) |
| **Regularizer Weights** | $\beta=0.001$ (1e-4–1e-1) | Weight decay: 0.0 | $\beta=0.001$ | $\lambda_3=0.1, \lambda_4=0.01$ | D grad-clip: 5.0 |
| **Baseline Epochs** | 15 | 20 | 15 | 5 warmup + 15 fine | 25 |
| **Optuna Trial Epochs**| 15 | 15 | 15 | 12 | 15 |
| **Full Retrain Epochs**| 60 | 35 | 50 | 25 | 100 |

---

## 5. Architectural Decisions & Research Rationale

1. **VAE over Plain AE (Tasks 1 & 2b)**:
   - *Rationale*: A variational latent space provides smooth interpolation and continuous generative priors for unseen degradation severities (Prakash et al., 2021; Soh & Cho, 2021).
   - *Loss formulation*: $\mathcal{L} = \alpha \cdot \text{L1} + (1-\alpha) \cdot (1-\text{SSIM}) + \beta \cdot \text{KL}$. Small $\beta=0.001$ prevents posterior collapse while prioritizing reconstruction.
2. **GroupNorm over BatchNorm**:
   - *Rationale*: Batch sizes vary across devices (16–64). BatchNorm causes unstable running statistics at small batch sizes; GroupNorm with 8 groups normalizes per-sample channels independently of batch size.
3. **Upsampling + Conv over Transpose-Conv**:
   - *Rationale*: Bilinear upsampling followed by standard 3×3 convolutions avoids the high-frequency checkerboard artifacts commonly introduced by `ConvTranspose2d` (Odena et al., 2016).
4. **Limited / Ablated Skip Connections**:
   - *Rationale*: Mao et al. (RED-Net) showed skip connections preserve high-frequency edges. However, autoencoders must retain a bottleneck. We baseline without skips and provide a single high-resolution skip connection ablation in `task1_vae_optuna.ipynb`.
5. **Soft Mixture-of-Experts (Task 3)**:
   - *Rationale*: Rather than hard routing (argmax), temperature-scaled softmax enables soft blending for boundary or mixed degradations (Ye et al., DAN-Net; Dong et al., PhyDAE).
   - *Load Balancing Regularizer*: $\lambda_4 \cdot \text{Var}(\text{usage}) / (\text{Mean}(\text{usage})^2)$ prevents routing collapse to a single expert.

---

## 6. Checkpoint Management & Multi-Device Workflow

- **Checkpoint Files**:
  - `*_latest.pt`: Saved every epoch (for resume upon crash / timeout).
  - `*_best.pt`: Saved only when validation loss improves.
- **Manifest Tracking**:
  - Every save automatically appends to [`research/checkpoints_manifest.json`](file:///run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/checkpoints_manifest.json) recording filename, epoch, validation loss, git commit hash, device type, and timestamp.
- **Multi-Device Workflow (Laptop $\to$ Colab $\to$ Uni PC)**:
  1. *Laptop (CPU)*: Set `TINY_RUN = True`, run 2 epochs to verify data loading, loss convergence, and checkpoint saving without errors.
  2. *Cloud / University PC (GPU)*: Set `TINY_RUN = False`, run full baseline or Optuna search.
  3. *Syncing*: Commit code and small SQLite `.db` study files to git; store `.pt` checkpoints via Git LFS or Hugging Face Hub as per Plan 4.
