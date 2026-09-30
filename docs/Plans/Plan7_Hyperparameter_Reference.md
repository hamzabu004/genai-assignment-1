# Plan 7 — Hyperparameter & Training Config Reference

Concrete baseline values + Optuna search ranges for all 4 tasks. Use this alongside Plan 6's per-task checklists — Plan 6 tells you *what step* to do, this tells you *what number* to put in it.

---

## Preprocessing ranges (fix this before anything else — differs by task)

| | Tasks 1, 2a, 2b, 3 (VAE/AE/classifier) | Task 4 (GAN) |
|---|---|---|
| Pixel normalization | **[0, 1]** (`/255` only) | **[-1, 1]** (`/127.5 - 1`) |
| Decoder/generator final activation | **sigmoid** | **tanh** |
| Why | SSIM's standard formula assumes a known, bounded `data_range` (typically 1.0) — keeping images in [0,1] avoids fudging that constant | Standard pix2pix/GAN convention; tanh output pairs naturally with [-1,1] targets |

Two separate preprocessing configs are needed — don't share one normalization constant across all 4 tasks.

---

## Tasks 1 & 2b — Universal VAE + 3 Specialist Autoencoders

| Hyperparameter | Baseline value | Optuna range |
|---|---|---|
| Batch size | 32 | categorical {16, 32, 64} |
| Learning rate | 1e-3 | log-uniform 1e-4 – 1e-3 |
| Optimizer | Adam (β1=0.9, β2=0.999) | fixed |
| Dropout | 0.1 | 0.0 – 0.3 |
| Latent/bottleneck dim | 128 | {64, 128, 256} |
| α (L1 vs SSIM weight) | 0.8 | 0.5 – 0.95 |
| β (KL weight, VAE only) | 0.001 | log-uniform 1e-4 – 1e-1 |
| Epochs — baseline sanity | 10–15 | — |
| Epochs — Optuna trial | 15–25 (~20% of full) | — |
| Epochs — full retrain | 60–100, early-stop patience ~10 on val loss | — |

Batch size is a convergence choice more than a memory constraint here — at 128×128, even a free-tier Colab GPU comfortably handles batch 64+. Dataset size (~5,900 training images) limits how many large-batch steps are useful.

## Task 2a — Corruption Classifier

| Hyperparameter | Baseline value | Optuna range |
|---|---|---|
| Batch size | 64 | categorical {32, 64, 128} |
| Learning rate | 1e-3 | log-uniform 1e-4 – 1e-2 |
| Weight decay | 0 | log-uniform 1e-5 – 1e-3 |
| Dropout | 0.2 | 0.0 – 0.5 |
| Epochs | 30–50, early-stop on val macro-F1 | — |

## Task 3 — Soft Mixture-of-Experts

| Hyperparameter | Baseline value | Optuna range |
|---|---|---|
| Batch size | 32 | {16, 32, 64} |
| Warm-up LR (gate only) | 1e-3 | fixed — short stage, not usually tuned |
| Joint fine-tune LR | 1e-4 | log-uniform 1e-5 – 1e-4 (must be smaller than warm-up) |
| Temperature τ | 1.0 | 0.5 – 2.0 |
| λ3 (classification weight) | 0.1 | 0.01 – 0.5 |
| λ4 (balance weight) | 0.01 | log-uniform 1e-3 – 1e-1 |
| Warm-up epochs | 5–10 | — |
| Joint fine-tune epochs (trial) | 20–40 | — |

## Task 4 — Conditional GAN (Face-to-Sketch)

| Hyperparameter | Baseline value | Optuna range |
|---|---|---|
| Batch size | 8 | categorical {4, 8, 16} — FS2K train split is only ~1,500–1,800 pairs |
| G learning rate | 2e-4 | log-uniform 1e-5 – 1e-3 |
| D learning rate | 2e-4 | log-uniform 1e-5 – 1e-3 (search independently of G) |
| Adam betas | **β1=0.5, β2=0.999** | fixed — standard GAN convention, differs from Tasks 1–3 |
| Base channels (ngf/ndf) | 64 | {32, 64, 128} |
| Style-embedding dim | 8 | {4, 8, 16, 32} |
| λ_L1 | 100 | 10 – 200 |
| Epochs — baseline sanity | ~20 (just confirm sketch-like output) | — |
| Epochs — Optuna trial | 30–50% of full budget (GANs need more steps to reveal instability) | — |
| Epochs — full retrain | 100–200; judge convergence via logged fixed-validation samples, not loss alone | — |

Add gradient clipping for the discriminator specifically (`clip_grad_norm_`, max-norm 1.0–5.0) — GAN training is the one place in this project where exploding gradients are a real risk.

---

## Cross-Cutting Settings (all tasks)
- `num_workers=2–4`, `persistent_workers=True` in every DataLoader — reduce if the University PC has fewer CPU cores
- Mixed precision (AMP) for speed on GPU, but compute SSIM/loss in fp32 even under `autocast` — SSIM is numerically sensitive in fp16
- Seed fixed to 42 everywhere (matches the data-split seed)
- Run a "double batch size until OOM" probe once on Colab and once on the University PC before finalizing the Optuna batch-size categorical range — don't guess VRAM, measure it. Set the range to what's safe on the *weaker* of the two GPU devices (laptop is CPU-only sanity checking only, per Plan 4, and isn't part of this ceiling)
