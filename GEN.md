# Master Experimentation, Manifest, and System Architecture Summary (GEN.md)

This document is the comprehensive, report-ready technical manifest for the Generative AI Image Restoration and Synthesis project. It synthesizes all empirical experimentation, mathematical formulations, model architectures, hyperparameter optimization studies, evaluation benchmarks, full-stack software architecture (FastAPI backend and Next.js frontend), ONNX deployment validation, and IEEE technical report requirements.

To maintain strict scientific integrity, this manifest explicitly separates **standalone exploratory runs** (single-corruption autoencoders, bottleneck investigations, and isolated routing studies) from **canonical deployment pipelines** (universal multi-corruption DAE, integrated hard router, joint soft MoE, and sharp conditional GAN).

---

## 1. Reproducible Data Protocol and Official Manifests

### 1.1 Oxford-IIIT Pet Dataset (Tasks 1--3)
Tasks 1, 2, and 3 use the Oxford-IIIT Pet dataset (37 cat and dog breeds). The original images serve as ground-truth clean targets.
- **Preprocessing Pipeline**: Images are converted to 3-channel RGB, resized to $128 \times 128$ using bilinear interpolation, and normalized to float32 values in $[0.0, 1.0]$.
- **Data Splitting**: Split using deterministic random seed `42` via [`research/generate_official_manifests.py`](file:///home/zaifi/genai-assignment-1/research/generate_official_manifests.py):
  - **Development Collection** (`trainval`): 80% Training ($N = 2,944$), 20% Validation ($N = 736$).
  - **Official Held-Out Test Collection** (`test`): $N = 3,669$ images, kept untouched during training and Optuna tuning.

| Manifest Name | File Path | Partition / Role | Fixed Sample Properties |
|---|---|---|---|
| **Development Split** | [`research/split_manifest_official.json`](file:///home/zaifi/genai-assignment-1/research/split_manifest_official.json) | Train (2,944) / Val (736) | Fixed image IDs partitioned with seed 42 |
| **Validation Manifest** | [`research/val_manifest_official.json`](file:///home/zaifi/genai-assignment-1/research/val_manifest_official.json) | Validation ($N = 736$) | 1 deterministic corruption condition, seed, and sampled parameters per image |
| **Test Benchmark** | [`research/test_manifest_official.json`](file:///home/zaifi/genai-assignment-1/research/test_manifest_official.json) | Held-Out Test ($N = 3,669$) | Clean + 9 corruptions (3 types $\times$ 3 severities) = **36,690 evaluation cells** |
| **Checkpoint Provenance** | [`research/checkpoints_manifest.json`](file:///home/zaifi/genai-assignment-1/research/checkpoints_manifest.json) | All Phases & Tasks | Epoch, val loss, device, timestamp, and Git commit hash |

### 1.2 Programmatic Corruption Specifications
- **Dynamic Training Corruptions**: Sampled uniformly at runtime ($p = 0.25$ per class: Clean, Salt-and-Pepper, Gaussian Blur, Rectangular Occlusion).
- **Deterministic Test Severities**: Parameterized exactly per assignment requirements:

| Corruption Class | Low Severity | Medium Severity | High Severity | Runtime Training Range |
|---|---|---|---|---|
| **Clean** | Pass-through ($x$) | Pass-through ($x$) | Pass-through ($x$) | Uncorrupted ($p=0.25$) |
| **Salt-and-Pepper** | $p = 0.03$ | $p = 0.08$ | $p = 0.15$ | $p \sim \mathcal{U}(0.02, 0.15)$, equal 0/1 |
| **Gaussian Blur** | $k = 3, \sigma = 0.7$ | $k = 5, \sigma = 1.5$ | $k = 7, \sigma = 2.5$ | $k \in \{3, 5, 7\}, \sigma \sim \mathcal{U}(0.5, 2.5)$ |
| **Rectangular Occlusion** | 1 rect, $\sim 10\%$ area | 2 rects, $\sim 20\%$ area | 3 rects, $\sim 35\%$ area | 1--3 black rects, $10\%$--$35\%$ joint area |

The test manifest records explicit box bounding coordinates `[x1, y1, x2, y2]`, guaranteeing byte-for-byte reproducibility across evaluation runs.

### 1.3 FS2K Facial Sketch Synthesis Dataset (Task 4)
- **Dataset Properties**: 2,104 high-resolution photo-sketch pairs across 3 distinct drawing styles (Style 1, Style 2, Style 3).
- **Data Splitting**: Official split (`anno_train.json` with 1,704 pairs, `anno_test.json` with 400 pairs).
- **Stratified Validation Split**: Seed `42`, stratified 85/15 split of official training data preserving style balance ($N_{train} = 1,448$, $N_{val} = 256$, $N_{test} = 400$).
- **Normalization**: Photos and paired sketches are scaled to $128 \times 128$ and normalized to $[-1.0, 1.0]$. Identical spatial transforms (random horizontal flips) are applied synchronously to both input photo and target sketch to preserve pixel-level correspondence.

---

## 2. Mathematical Foundations & Loss Formulations

```
Task 1: Universal DAE         Task 2: Hard Router           Task 3: Soft MoE            Task 4: Conditional GAN
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐    ┌─────────────────────────┐
│     Corrupted x̃       │    │     Corrupted x̃       │    │     Corrupted x̃       │    │   Photo x + Style s     │
│          │            │    │    ┌───┴───┐          │    │    ┌───┴───┐          │    │            │            │
│       ConvDAE         │    │ Classifier │          │    │ Gate g(x̃) │          │    │      UNetGenerator      │
│     (Bottleneck)      │    │    └───┬───┘          │    │    └───┬───┘          │    │   (FiLM + Multi-Scale)  │
│          │            │    │     k̂=argmax          │    │    w = Softmax(g/τ)   │    │            │            │
│   Reconstruction x̂    │    │ ┌──────┴──────┐       │    │ ┌──────┴──────┐       │    │    Synthesized Sketch ŷ │
│          │            │    │Pass   D_salt  D_blur  │    │Pass   D_salt  D_blur  │    │            │            │
│   L_AE = α·L1+(1-α)SSIM│    │          │            │    │   w_c·x̃ + Σ w_k·D_k   │    │  Two-Scale PatchGAN (D) │
└───────────────────────┘    └───────────────────────┘    └───────────────────────┘    └─────────────────────────┘
```

### 2.1 Structural Similarity Index Measure (SSIM)
Implemented in [`research/src/losses.py`](file:///home/zaifi/genai-assignment-1/research/src/losses.py#L24-L70) as a differentiable PyTorch operator. Using an $11 \times 11$ Gaussian kernel with standard deviation $\sigma = 1.5$:
$$\mathrm{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$
Where $C_1 = (0.01 \cdot L)^2$ and $C_2 = (0.03 \cdot L)^2$, with dynamic range $L = 1.0$ for restoration tasks and $L = 2.0$ for GAN tasks in $[-1, 1]$.

### 2.2 Autoencoder Reconstruction Loss (Tasks 1 & 2b)
$$\mathcal{L}_{AE}(x, \hat{x}) = \alpha\,\mathcal{L}_{L1}(x, \hat{x}) + (1-\alpha)\,\big(1 - \mathrm{SSIM}(x, \hat{x})\big)$$
- **$\mathcal{L}_{L1}(x, \hat{x}) = \frac{1}{CHW}\sum |x - \hat{x}|$**: Enforces sharp pixel-wise convergence without blurriness induced by $L2$.
- **$1 - \mathrm{SSIM}(x, \hat{x})$**: Enforces structural, edge, and contrast fidelity.
- **$\alpha$**: Weight parameter tuned dynamically via Optuna ($\alpha \in [0.5, 0.95]$).

### 2.3 Corruption Classifier Loss (Task 2a)
Multiclass cross-entropy loss trained over balanced batches ($B = 64$):
$$\mathcal{L}_{CE}(y, p) = -\sum_{k=1}^4 y_k \log p_k, \quad p = \mathrm{softmax}(f(x))$$

### 2.4 Soft Mixture-of-Experts Joint Loss (Task 3)
$$\mathcal{L}_{joint} = \lambda_1\,\mathcal{L}_{L1}(x, \hat{x}) + \lambda_2\,\big(1 - \mathrm{SSIM}(x, \hat{x})\big) + \lambda_3\,\mathcal{L}_{CE}(y, g(\tilde{x})) + \lambda_4\,\mathcal{L}_{balance}(w)$$
Where the temperature-controlled gating distribution is:
$$w_k = \frac{\exp(g_k(\tilde{x}) / \tau)}{\sum_{j=1}^4 \exp(g_j(\tilde{x}) / \tau)}, \quad k \in \{\text{clean}, \text{salt}, \text{blur}, \text{occlusion}\}$$
The load balancing regularizer penalizes deviation from uniform expert allocation across balanced mini-batches:
$$\mathcal{L}_{balance}(w) = \sum_{k=1}^4 \left( \bar{w}_k - \frac{1}{4} \right)^2, \quad \bar{w}_k = \frac{1}{B}\sum_{i=1}^B w_{i,k}$$

### 2.5 Style-Conditioned GAN Objective (Task 4)
- **Two-Scale PatchGAN Discriminator Loss**:
$$\mathcal{L}_D = \frac{1}{2} \mathbb{E}_{x, y, s}\big[\log D(x, y, s)\big] + \frac{1}{2} \mathbb{E}_{x, s}\big[\log(1 - D(x, G(x, s), s))\big]$$
Averaged across full scale ($128 \times 128$) and half scale ($64 \times 64$).
- **Generator Loss**:
$$\mathcal{L}_G = \mathcal{L}_{adv}(G) + \lambda_{L1}\mathcal{L}_{L1}(y, \hat{y}) + \lambda_{edge}\mathcal{L}_{edge}(y, \hat{y}) + \lambda_{SSIM}(1 - \mathrm{SSIM}(y, \hat{y}))$$
Where $\mathcal{L}_{edge}$ computes the $L1$ difference between horizontal and vertical Sobel gradient responses:
$$K_x = \frac{1}{8}\begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = K_x^T$$

---

## 3. Taxonomy: Standalone vs Canonical Combined Experiments

| Dimension | Standalone Exploratory Experiments | Canonical Combined Pipeline (Final Deployment) |
|---|---|---|
| **Role in Project** | Hypothesis testing, capacity sweeps, skip connection ablation, failure baseline discovery | Integrated production system, multi-task compatibility, unified ONNX exports |
| **Model Scope** | Isolated per-corruption DAEs, non-unified classifier / specialist studies | Universal DAE, official hard router, joint Soft MoE, multi-scale conditional GAN |
| **Artifacts Produced** | `dae_*.pt`, `task1_standalone_*.pt`, `task2_standalone_*.pt` | `task1_universal_ae_final_best.pt`, `task2a_classifier_final_best.pt`, etc. |
| **Storage / Evidence** | `research/reports/task1_standalone_test_metrics.json`, `task2_standalone_metrics.json` | `export/reports/final_best_onnx_validation.json`, `task1_universal_ae_validation.json` |
| **Report Status** | Section on architectural exploration, ablations, and baseline trade-offs | Primary quantitative results tables in IEEE report |

---

## 4. Standalone Empirical Experiments & Architecture Ablations

### 4.1 Skip Connection & Upsampling Method Ablation Study
To verify the assignment requirement that restoration models possess a genuine compressed bottleneck rather than bypassing features via trivial skips, two architectures were compared on Oxford-IIIT Pet under 10% salt-and-pepper noise:
- **Baseline with Skip Connections** ([`dae_testing.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing.ipynb)): Encoder downsamples to $32 \times 32$; 16-channel bottleneck; decoder uses skip connections at $64 \times 64$ and $128 \times 128$.
- **Strict Bottleneck with Transposed Convolutions** ([`dae_testing_no_skip_bottleneck32_transpose.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing_no_skip_bottleneck32_transpose.ipynb)): 3-stage encoder downsampling to $16 \times 16 \times 256$, projected to a 32-channel bottleneck with **no skip connections**, decoded via `ConvTranspose2d`.

| Noise Density | Skip Baseline PSNR (dB) | Skip Baseline SSIM | Strict Bottleneck PSNR (dB) | Strict Bottleneck SSIM | $\Delta$ PSNR (Skip Advantage) |
|---|---:|---:|---:|---:|---:|
| **2%** | 36.51 | 0.972 | 24.16 | 0.887 | +12.35 dB |
| **5%** | 35.77 | 0.964 | 24.16 | 0.885 | +11.61 dB |
| **10% (Nominal)** | **34.33** | **0.9465** | **24.12** | **0.8815** | **+10.21 dB** |
| **20%** | 31.08 | 0.890 | 23.94 | 0.866 | +7.14 dB |
| **30%** | 27.44 | 0.797 | 23.45 | 0.831 | +3.99 dB |
| **50%** | 19.36 | 0.501 | 20.26 | 0.644 | -0.90 dB |

> **Key Architectural Insight for IEEE Report**:
> While unrestricted skip connections achieve superior PSNR (+10.21 dB at nominal noise) by passing high-frequency background details directly to the decoder, they allow the network to evade learning semantic image representations. Furthermore, transposed convolutions generated noticeable checkerboard artifacts in reconstructed images. Consequently, the canonical architecture adopted **bilinear interpolation + convolution** and an **explicit 1D latent bottleneck projection** without high-resolution skip connections.

### 4.2 Bottleneck Dimension Optimization Study
Investigated in [`dae_optuna_bottleneck.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_optuna_bottleneck.ipynb) and [`dae_saltpepper_bottleneck.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_saltpepper_bottleneck.db) across 20 Optuna trials (9 completed, 11 pruned):

| Bottleneck Channels | Completed Trials | Best Validation Loss | Best Trial Learning Rate | Test Evaluation (10% Salt) |
|---|---:|---:|---:|---|
| **8 channels** | 2 | 0.542996 | 0.001332 | Severe underfitting, loss of fine texture |
| **16 channels** | 2 | 0.548321 | 0.000185 | Blur on facial contours |
| **32 channels (Winner)** | **4** | **0.541155** (Trial #12) | **0.001005** | **PSNR: 14.91 $\rightarrow$ 24.82 dB (+9.91 dB), SSIM: 0.6884** |
| **64 channels** | 1 | 0.541267 | 0.000711 | Marginal gain (+0.0001 val loss) for double parameters |

### 4.3 Isolated Single-Corruption Hyperparameter Searches
- **Salt-and-Pepper Autoencoder** ([`dae_saltpepper_simple_optuna.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_saltpepper_simple_optuna.db) & [`dae_testing_salt_peper_simple.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing_salt_peper_simple.ipynb)):
  - 20 trials (13 complete, 7 pruned). Best trial #19: `val_loss = 0.036348`, `batch_size = 16`, `base_channels = 64`, `bottleneck_channels = 64`, `lr = 0.0009466`, `alpha = 0.9014`.
  - Winning retrain (30 epochs): best val loss `0.0307` at epoch 29. Held-out test performance at 10% noise: **PSNR 14.91 $\rightarrow$ 31.58 dB (+16.67 dB)**, **SSIM 0.2647 $\rightarrow$ 0.9171 (+0.6524)**.
- **Gaussian Blur Autoencoder** ([`dae_gaussian_blur_optuna.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_gaussian_blur_optuna.db) & [`dae_testing_gaussian_blur_optuna.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing_gaussian_blur_optuna.ipynb)):
  - 11 trials (6 complete). Best trial #8: `val_loss = 0.048056`, `batch_size = 16`, `base_channels = 64`, `bottleneck_channels = 96`, `lr = 0.0004038`, `alpha = 0.8644`.

### 4.4 Standalone Task 1 Universal Multi-Corruption Benchmark
Recorded in [`research/reports/task1_standalone_test_metrics.json`](file:///home/zaifi/genai-assignment-1/research/reports/task1_standalone_test_metrics.json). Model trained on all 4 conditions simultaneously:
- **Selected Hyperparameters**: `lr = 0.00014301`, `batch_size = 16`, `latent_dim = 512`, `base_channels = 32`, `dropout = 0.1685`, `alpha = 0.9345`, validation objective `0.111986`.
- **Complete Test Benchmark ($N = 3,669$ per condition, 36,690 total evaluations)**:

| True Condition | Low Severity (PSNR / SSIM) | Medium Severity (PSNR / SSIM) | High Severity (PSNR / SSIM) |
|---|---|---|---|
| **Clean** | 19.087 dB / 0.4617 | 19.087 dB / 0.4617 | 19.087 dB / 0.4617 |
| **Salt-and-Pepper** | 19.100 dB / 0.4615 | 19.067 dB / 0.4607 | 18.959 dB / 0.4579 |
| **Gaussian Blur** | 19.118 dB / 0.4617 | 19.157 dB / 0.4612 | 19.168 dB / 0.4595 |
| **Occlusion** | 18.436 dB / 0.4494 | 17.993 dB / 0.4389 | 17.363 dB / 0.4245 |

### 4.5 Standalone Task 2 Hard Routing & Failure Mode Audit
Evaluated on full 36,690-sample test set in [`research/reports/task2_standalone_metrics.json`](file:///home/zaifi/genai-assignment-1/research/reports/task2_standalone_metrics.json) & [`research/notebooks/task2_standalone.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/task2_standalone.ipynb):

#### 4.5.1 Standalone Classifier Results
- **Overall Accuracy**: **99.733%** (36,592 correct / 36,690 decisions, only 98 routing errors).
- **Macro-Averaged Metrics**: Precision: `0.99564`, Recall: `0.99573`, F1-Score: `0.99569`.
- **Per-Class F1-Scores**: Clean: `0.9875`, Salt: `0.9999`, Blur: `0.9975`, Occlusion: `0.9979`.
- **Full Confusion Matrix** (Rows: True Class, Columns: Predicted Class [Clean, Salt, Blur, Occlusion]):
$$\begin{bmatrix} 3624 & 1 & 17 & 27 \\ 0 & 11006 & 0 & 1 \\ 34 & 0 & 10968 & 5 \\ 13 & 0 & 0 & 10994 \end{bmatrix}$$

#### 4.5.2 Oracle vs. Predicted Routing Comparison

| Condition | Low Severity: Oracle / Predicted | Medium Severity: Oracle / Predicted | High Severity: Oracle / Predicted |
|---|---|---|---|
| **Clean** | 100.0 dB / 98.967 dB (SSIM 1.0 / 0.9920) | 100.0 dB / 98.967 dB (SSIM 1.0 / 0.9920) | 100.0 dB / 98.967 dB (SSIM 1.0 / 0.9920) |
| **Salt** | 18.039 dB / 18.038 dB (SSIM 0.4204 / 0.4204) | 18.029 dB / 18.029 dB (SSIM 0.4203 / 0.4203) | 18.005 dB / 18.005 dB (SSIM 0.4199 / 0.4199) |
| **Blur** | 18.117 dB / 18.215 dB (SSIM 0.4226 / 0.4276) | 18.154 dB / 18.153 dB (SSIM 0.4222 / 0.4222) | 18.170 dB / 18.170 dB (SSIM 0.4214 / 0.4214) |
| **Occlusion** | 16.877 dB / 16.908 dB (SSIM 0.4002 / 0.4024) | 16.742 dB / 16.742 dB (SSIM 0.3974 / 0.3974) | 16.394 dB / 16.394 dB (SSIM 0.3912 / 0.3912) |

#### 4.5.3 Severe Failure Cases: The Clean Misrouting Catastrophe
When a clean image is correctly routed to the identity bypass, PSNR is $\infty$ (clamped to 100 dB) and SSIM is 1.0. However, when the classifier erroneously misclassifies a clean image as occluded or blurred, it is routed to a destructive specialist, causing a catastrophic quality drop:
1. `chihuahua_84.jpg` (Clean): Predicted Occlusion ($p_{occl} = 0.9999$). Oracle PSNR: 100.00 dB $\rightarrow$ Routed PSNR: 8.87 dB (**Drop: 91.13 dB**, SSIM: 0.3750).
2. `samoyed_72.jpg` (Clean): Predicted Occlusion ($p_{occl} = 0.9794$). Oracle PSNR: 100.00 dB $\rightarrow$ Routed PSNR: 9.23 dB (**Drop: 90.77 dB**, SSIM: 0.3749).
3. `shiba_inu_95.jpg` (Clean): Predicted Occlusion. Oracle PSNR: 100.00 dB $\rightarrow$ Routed PSNR: 9.83 dB (**Drop: 90.17 dB**).
4. `german_shorthaired_7.jpg` (Clean): Predicted Blur ($p_{blur} = 0.9106$). Oracle PSNR: 100.00 dB $\rightarrow$ Routed PSNR: 14.15 dB (**Drop: 85.85 dB**, SSIM: 0.1470).

---

## 5. Canonical Combined Pipeline Experiments (Deployment Artifacts)

### 5.1 Task 1: Universal Multi-Corruption DAE
- **Architecture**: 5-stage convolutional encoder ($128 \rightarrow 4 \times 4$), GroupNorm (8 groups), linear projection to 256-dim latent vector, bilinear upsampling + convolution decoder. Parameter count: **15,106,307**.
- **Optuna Tuning** ([`research/configs/task1_best_config.yaml`](file:///home/zaifi/genai-assignment-1/research/configs/task1_best_config.yaml)):
  - 25 trials exploring learning rate ($10^{-4}$--$10^{-3}$), batch size (16, 32, 64), latent dimension (64, 128, 256), base channels (32, 64), dropout (0--0.3), and $\alpha$ (0.5--0.95).
  - **Winning Trial (#12)**: Validation Objective = `0.126499`, `lr = 0.00042373`, `batch_size = 16`, `latent_dim = 256`, `base_channels = 64`, `dropout = 0.041506`, `alpha = 0.949093`.
- **60-Epoch Retraining**:
  - `task1_universal_ae_final_best.pt`: Selected at **Epoch 59**, Validation Loss = **0.098954**.
  - `task1_universal_ae_final_latest.pt`: Epoch 60, Validation Loss = 0.106622.
- **Validation Benchmark** ([`export/reports/task1_universal_ae_validation.json`](file:///home/zaifi/genai-assignment-1/export/reports/task1_universal_ae_validation.json), $N = 736$):

| Condition | Sample Count ($n$) | Mean Squared Error (MSE) | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| **Clean** | 183 | 0.012371 | 19.548 | 0.5255 |
| **Salt-and-Pepper** | 179 | 0.012219 | 19.567 | 0.5117 |
| **Gaussian Blur** | 178 | 0.012593 | 19.500 | 0.5239 |
| **Occlusion** | 196 | 0.015669 | 18.572 | 0.5089 |

### 5.2 Task 2: Hard-Routed Specialists

#### 5.2.1 Task 2a: Corruption Classifier
- **Architecture**: 4-stage convolutional downsampler ($128 \rightarrow 8 \times 8$), GroupNorm, Global Average Pooling, 128-dim hidden MLP head. Parameter count: **190,548**.
- **Optuna Tuning** ([`research/notebooks/task2a_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task2a_study.db)):
  - 20 trials (10 complete, 10 pruned).
  - **Winning Trial (#18)**: Validation Accuracy = **99.05%**, `lr = 0.00061523`, `batch_size = 64`, `base_channels = 16`, `dropout = 0.001970`, `weight_decay = 0.00017498`.
- **35-Epoch Retraining**:
  - `task2a_classifier_final_best.pt`: Selected at **Epoch 33**, Validation Loss = **0.037689**, Validation Accuracy = **99.18%**.

#### 5.2.2 Task 2b: Specialist Autoencoders
- **Optuna Architecture Search** ([`research/notebooks/task2b_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task2b_study.db)):
  - 53 trials (21 complete, 32 pruned).
  - **Winning Architecture (#30)**: Validation Objective = `0.108699`, `lr = 0.00010677`, `batch_size = 32`, `latent_dim = 256`, `base_channels = 64`, `dropout = 0.026026`, `alpha = 0.947144`.
- **35-Epoch Independent Retraining**:
  - **Salt Specialist** (`task2b_specialist_salt_final_best.pt`): Selected at **Epoch 32**, Validation Loss = **0.151279**.
  - **Blur Specialist** (`task2b_specialist_blur_final_best.pt`): Selected at **Epoch 34**, Validation Loss = **0.145533**.
  - **Occlusion Specialist** (`task2b_specialist_occlusion_final_best.pt`): Selected at **Epoch 35**, Validation Loss = **0.169160**.

### 5.3 Task 3: Jointly Trained Soft Mixture-of-Experts
- **Initialization**: Gating head initialized from `task2a_classifier_final_best.pt`; 3 specialist branches initialized from Task 2b final checkpoints.
- **Warm-Up Phase (Epochs 1--5)**: Specialist weights frozen; gate trained with cross-entropy and balance regularization.
  - Gate usage distribution across warm-up: Clean: `~0.58`, Salt: `~0.18`, Blur: `~0.10`, Occlusion: `~0.14`. Validation loss stabilized at `0.1935`.
- **Optuna Hyperparameter Tuning** ([`research/notebooks/task3_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task3_study.db)):
  - 15 trials (11 complete, 4 pruned).
  - **Winning Trial (#12)**: Validation Loss = `0.132747`, `finetune_lr = 8.2205e-5`, `temperature = 1.9062`, `lambda3 = 0.220089`, `lambda4 = 0.002200`, `alpha = 0.928536`.
- **25 Joint Fine-Tuning Epochs**:
  - Progressive validation loss decrease: Epoch 1 (`0.1786`, SSIM 0.6087) $\rightarrow$ Epoch 10 (`0.1513`) $\rightarrow$ Epoch 20 (`0.1215`) $\rightarrow$ **Epoch 25 Best (`0.115984`, SSIM 0.6398)**.
  - Checkpoint: `task3_soft_moe_final_best.pt`.

### 5.4 Task 4: Style-Conditioned Face-to-Sketch Conditional GAN
- **Generator**: U-Net encoder-decoder with style embedding (dim = 16) injected into bottleneck and intermediate decoder stages via FiLM modulation. Bilinear upsampling and grayscale constraint ($R=G=B$) to eradicate chromatic aberration.
- **Discriminator**: Two-scale PatchGAN evaluating $70 \times 70$ receptive field patches at $128 \times 128$ and $64 \times 64$ scales.
- **Optuna Tuning** ([`research/notebooks/task4_sharp_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task4_sharp_study.db)):
  - 35 trials (30 completed).
  - **Winning Trial (#28)**: Validation Score = **0.333974**, `g_lr = 0.00017386`, `d_lr = 1.2870e-5`, `batch_size = 8`, `base_channels = 32`, `dropout = 0.071872`, `embed_dim = 16`, `lambda_l1 = 161.08`, `lambda_edge = 28.61`, `lambda_ssim = 13.67`, `residual_refinement = False`.
- **Retraining History**:
  - `task4_generator_final_best.pt`: Selected at **Epoch 18**, Validation Loss = **0.342283**.
  - `task4_generator_final_latest.pt`: Epoch 60, Validation Loss = 0.361942.

---

## 6. End-to-End ONNX Export & Parity Verification

All 7 production models were exported from PyTorch to ONNX (Opset 17) and validated using ONNX Runtime with tolerances $rtol = 10^{-4}, atol = 10^{-4}$. Verification results from [`export/reports/final_best_onnx_validation.json`](file:///home/zaifi/genai-assignment-1/export/reports/final_best_onnx_validation.json):

| Model Key | Source Checkpoint | ONNX Filename | Size (Bytes) | SHA-256 Checksum | Max Absolute Parity Error | I/O Tensor Signatures |
|---|---|---|---:|---|---:|---|
| **Universal DAE** | `task1_universal_ae_final_best.pt` | `task1_universal_ae.onnx` | 60,444,166 | `ce30b787...` | $4.17 \times 10^{-7}$ | `input`: [B,3,128,128] $\rightarrow$ `output`: [B,3,128,128] |
| **Classifier** | `task2a_classifier_final_best.pt` | `task2_classifier.onnx` | 769,623 | `43422689...` | $9.54 \times 10^{-7}$ | `input`: [B,3,128,128] $\rightarrow$ `logits`: [B,4] |
| **Salt Specialist** | `task2b_specialist_salt_final_best.pt` | `task2_specialist_salt.onnx` | 60,444,166 | `381aeea7...` | $3.87 \times 10^{-7}$ | `input`: [B,3,128,128] $\rightarrow$ `output`: [B,3,128,128] |
| **Blur Specialist** | `task2b_specialist_blur_final_best.pt` | `task2_specialist_blur.onnx` | 60,444,166 | `3b2ea481...` | $2.98 \times 10^{-7}$ | `input`: [B,3,128,128] $\rightarrow$ `output`: [B,3,128,128] |
| **Occl Specialist** | `task2b_specialist_occlusion_final_best.pt` | `task2_specialist_occlusion.onnx` | 60,444,166 | `1ff2b800...` | $4.77 \times 10^{-7}$ | `input`: [B,3,128,128] $\rightarrow$ `output`: [B,3,128,128] |
| **Soft MoE** | `task3_soft_moe_final_best.pt` | `task3_soft_moe.onnx` | 48,369,274 | `1235b928...` | $2.86 \times 10^{-6}$ | `input`: [B,3,128,128] $\rightarrow$ `output`, `routing_probs`, `gate_logits` |
| **Face-to-Sketch** | `task4_generator_final_best.pt` | `task4_generator.onnx` | 12,720,870 | `dd8915e9...` | Style 0: $2.15 \times 10^{-6}$<br>Style 1: $2.59 \times 10^{-6}$<br>Style 2: $1.79 \times 10^{-6}$ | `photo`: [B,3,128,128], `style_idx`: [B] $\rightarrow$ `output`: [B,3,128,128] |

---

## 7. Backend Architecture & Service Design

The backend is built with **FastAPI** and **ONNX Runtime**, providing asynchronous, high-throughput model inference, robust input sanitization, and structured responses.

```
backend/
├── app/
│   ├── core/
│   │   ├── config.py             # Pydantic BaseSettings (model paths, CORS, thresholds)
│   │   └── logging.py            # Structured system logger
│   ├── main.py                   # FastAPI app factory, lifespan loader, global error handlers
│   ├── routers/
│   │   ├── health.py             # GET  /health (system health & loaded ONNX models)
│   │   ├── universal_restoration.py # POST /universal-restoration/infer
│   │   ├── hard_routing.py       # POST /hard-routing/infer
│   │   ├── soft_mixture.py       # POST /soft-mixture/infer
│   │   └── face_to_sketch.py     # POST /face-to-sketch/infer
│   ├── schemas/                  # Pydantic v2 data transfer objects
│   │   ├── common.py             # HealthResponse, ApiErrorResponse
│   │   ├── universal_restoration.py # UniversalRestorationResponse, CorruptionApplied
│   │   ├── hard_routing.py       # HardRoutingResponse
│   │   ├── soft_mixture.py       # SoftMixtureResponse
│   │   └── face_to_sketch.py     # FaceToSketchResponse
│   ├── services/
│   │   ├── onnx_runtime_manager.py # Session pool, hardware provider hierarchy (CUDA->CPU)
│   │   ├── preprocessing.py      # Resizing to 128x128, RGB convert, unit tensor normalizer
│   │   ├── postprocessing.py     # Tensor to Base64 PNG, jet colormap residual error maps
│   │   └── corruption.py         # Runtime & deterministic manifest corruption engines
│   └── utils/
│       └── timing.py             # Microsecond-accurate latency context manager
├── models/                       # Active deployed ONNX weight files (.onnx)
└── tests/                        # Pytest suite (6 test files)
```

### 7.1 Key Endpoints and Data Contracts
1. **`GET /health`**: Returns engine status and list of active loaded ONNX models.
2. **`POST /universal-restoration/infer`**: Accepts an uploaded image or sample filename, applies requested or manifest corruption, performs inference with `task1_universal_ae.onnx`, calculates PSNR/SSIM, generates a jet colormap error map, and returns base64 images with runtime latency.
3. **`POST /hard-routing/infer`**: Evaluates input with `task2_classifier.onnx`, extracts probabilities, selects expert via argmax, routes clean images through identity bypass, and returns class probabilities, selected branch, and reconstructed image.
4. **`POST /soft-mixture/infer`**: Evaluates input with `task3_soft_moe.onnx` with selectable temperature $\tau$, computes continuous gating weights, dominant expert, and weighted output blend.
5. **`POST /face-to-sketch/infer`**: Ingests portrait image and categorical style index ($0, 1, 2$), runs `task4_generator.onnx`, applies grayscale and contrast normalization, and returns synthetic sketch.

### 7.2 Verification & Test Suite
The backend is verified through an automated test suite ([`backend/tests/`](file:///home/zaifi/genai-assignment-1/backend/tests)):
- [`test_health.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_health.py): Health check liveness and model pool confirmation.
- [`test_universal_restoration.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_universal_restoration.py): Manifest sample retrieval, corruption application, PSNR/SSIM correctness, and error map validity.
- [`test_hard_routing.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_hard_routing.py): Argmax classification and identity bypass verification.
- [`test_soft_mixture.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_soft_mixture.py): Softmax gating summation ($\sum w_k = 1.0$) and temperature scaling.
- [`test_face_to_sketch.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_face_to_sketch.py): Style index binding and sketch dimension verification.
- [`test_error_handling.py`](file:///home/zaifi/genai-assignment-1/backend/tests/test_error_handling.py): Rejection of corrupted files, invalid MIME types, out-of-range severities, and path traversal attempts (`../../etc/passwd`).

---

## 8. Frontend Architecture & Workspace Design

The frontend is implemented in **Next.js 15 (App Router)**, **React 19**, and **Tailwind CSS**, designed in strict accordance with the Google Stitch UI specifications preserved in [`docs/design-handoff/stitch_ai_image_restoration_suite`](file:///home/zaifi/genai-assignment-1/docs/design-handoff/stitch_ai_image_restoration_suite).

```
frontend/
├── app/
│   ├── layout.tsx                # Shell layout with persistent sidebar & navigation
│   ├── page.tsx                  # Home redirect to /universal-restoration
│   ├── universal-restoration/    # Workspace 1: Universal Multi-Corruption Restoration
│   ├── hard-routing/             # Workspace 2: Corruption Classifier & Hard Specialists
│   ├── soft-mixture/             # Workspace 3: Differentiable Soft MoE with Temperature
│   └── face-to-sketch/           # Workspace 4: Style-Conditioned Face-to-Sketch Synthesis
├── components/
│   ├── layout/Sidebar.tsx        # Persistent navigation sidebar with active task pill
│   └── ui/
│       ├── UploadZone.tsx        # Drag-and-drop file upload, webcam modal, sample selector
│       ├── ImagePanel.tsx        # Side-by-side comparison, zoom, overlay slider, error map
│       ├── BarChart.tsx          # Real-time probability & gating distribution visualizer
│       ├── Pill.tsx              # Degradation status badges & selected expert indicator
│       ├── Button.tsx            # Primary action buttons with loading spinners
│       ├── ErrorBanner.tsx       # Standardized user-facing error callout
│       └── Loader.tsx            # Animated inference processing indicator
├── hooks/
│   └── useInference.ts           # Unified React hook managing loading, latency, errors, API calls
└── lib/
    ├── api.ts                    # Typed API client interfacing with FastAPI backend
    ├── types.ts                  # TypeScript interfaces matching backend Pydantic schemas
    └── sampleImages.ts           # Curated preloaded Oxford-IIIT Pet & portrait gallery
```

### 8.1 Four Dedicated Workspaces
1. **Universal Restoration (`/universal-restoration`)**:
   - Live corruption controls: Clean, Salt-and-Pepper, Gaussian Blur, Rectangular Occlusion.
   - Severity level selectors (Low, Medium, High).
   - Side-by-side target, corrupted, restored images, and residual error map display.
   - Inference latency badge and real-time PSNR / SSIM readout.
2. **Hard-Routed Restoration (`/hard-routing`)**:
   - Displays 4-class classifier probability distribution via animated bar charts.
   - Visual indicator showing routing decision and activated specialist.
   - Highlighted identity pass-through badge when clean image is detected.
3. **Soft Mixture-of-Experts (`/soft-mixture`)**:
   - Interactive temperature slider ($\tau \in [0.1, 3.0]$) controlling gating entropy.
   - Soft expert weight horizontal bar chart showing individual contributions.
   - Dominant expert badge and composite blended output visualization.
4. **Face-to-Sketch Generator (`/face-to-sketch`)**:
   - Webcam capture integration and facial photo upload zone.
   - Style selectors for Style 1, Style 2, and Style 3 with thumbnail visual previews.
   - Post-processing toggles (contrast enhancement, sharpening).
   - Direct download button for high-resolution synthesized sketches.

---

## 9. Containerization & Deployment Orchestration

The application is containerized with multi-stage Docker builds orchestrated via [`docker-compose.yml`](file:///home/zaifi/genai-assignment-1/docker-compose.yml):

```yaml
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      VALIDATION_MANIFEST_PATH: /app/validation/val_manifest_official.json
      VALIDATION_IMAGE_DIR: /app/validation/images
    env_file:
      - ./backend/.env.example

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
```

- **One-Command Deployment**:
  ```bash
  docker compose up --build
  ```
- **Access Points**:
  - Web User Interface: `http://localhost:3000`
  - FastAPI Interactive Documentation (Swagger UI): `http://localhost:8000/docs`
  - Health & Model Liveness: `http://localhost:8000/health`

---

## 10. IEEE Technical Report Insertion Checklist & Manifest Map

| Report Section | Local Source Files & Artifacts | Primary Values / Evidence to Insert |
|---|---|---|
| **I. Introduction & System Architecture** | [`docs/GenAI_Assignment 1.md`](file:///home/zaifi/genai-assignment-1/docs/GenAI_Assignment%201.md), [`README.md`](file:///home/zaifi/genai-assignment-1/README.md) | Problem formulation across 4 tasks; Next.js/FastAPI full-stack containerized architecture |
| **II. Data Protocol & Manifests** | [`research/generate_official_manifests.py`](file:///home/zaifi/genai-assignment-1/research/generate_official_manifests.py), `split_manifest_official.json` | Seed 42, 80/20 train/val split (2,944 / 736); 3,669 test images; 36,690 deterministic test evaluation cells |
| **III. Loss Formulations** | [`research/src/losses.py`](file:///home/zaifi/genai-assignment-1/research/src/losses.py) | Mathematical equations: SSIM ($11\times 11, \sigma=1.5$), DAE composite ($\alpha \cdot L1 + (1-\alpha)SSIM$), MoE joint balance loss, Two-scale PatchGAN |
| **IV. Architecture Ablation & Standalone Studies** | [`dae_testing.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing.ipynb), [`dae_testing_no_skip_bottleneck32_transpose.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/dae_testing_no_skip_bottleneck32_transpose.ipynb), `task1_standalone_test_metrics.json` | Skip connection ablation table (34.33 vs 24.12 dB); checkerboard artifact analysis; bottleneck capacity study (32-ch winner); Standalone Task 1 10-condition table |
| **V. Task 1: Universal DAE** | [`task1_best_config.yaml`](file:///home/zaifi/genai-assignment-1/research/configs/task1_best_config.yaml), `task1_universal_ae_validation.json` | Optuna trial #12 (lr=4.24e-4, batch=16, latent=256, base=64, alpha=0.949); Epoch 59 best val loss (0.098954); Validation table (Clean 19.55 dB, Blur 19.50 dB, Salt 19.57 dB, Occl 18.57 dB) |
| **VI. Task 2: Hard-Routed Restoration** | [`task2a_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task2a_study.db), [`task2b_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task2b_study.db), `task2_standalone_metrics.json` | Classifier test accuracy 99.73%, macro F1 0.9957, full confusion matrix; Specialist validation losses (Salt 0.151, Blur 0.146, Occl 0.169); Oracle vs Predicted routing table; Clean misrouting catastrophe analysis (-91 dB drop) |
| **VII. Task 3: Soft Mixture-of-Experts** | [`task3_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task3_study.db), [`task3_moe_simple.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/task3_moe_simple.ipynb) | Warmup gate usage logging; Optuna trial #12 (finetune_lr=8.22e-5, temp=1.9062, lambda3=0.2201, lambda4=0.0022); 25-epoch joint curve (val loss 0.1786 $\rightarrow$ 0.1160, SSIM 0.6398) |
| **VIII. Task 4: Face-to-Sketch cGAN** | [`task4_sharp_study.db`](file:///home/zaifi/genai-assignment-1/research/notebooks/task4_sharp_study.db), [`task4_gan_simple.ipynb`](file:///home/zaifi/genai-assignment-1/research/notebooks/task4_gan_simple.ipynb) | Optuna trial #28 (g_lr=1.74e-4, d_lr=1.29e-5, batch=8, embed_dim=16, lambda_l1=161.08, lambda_edge=28.61); 60-epoch retrain (epoch 18 best val loss 0.342283); 25-epoch G/D loss tracking |
| **IX. ONNX Export & Parity Verification** | [`final_best_onnx_validation.json`](file:///home/zaifi/genai-assignment-1/export/reports/final_best_onnx_validation.json) | Full table of 7 models with byte counts, SHA-256 hashes, opset 17, and max absolute PyTorch-ONNX parity error ($< 2.86 \times 10^{-6}$) |
| **X. Full-Stack Software Product** | [`backend/`](file:///home/zaifi/genai-assignment-1/backend), [`frontend/`](file:///home/zaifi/genai-assignment-1/frontend), [`docs/design-handoff/`](file:///home/zaifi/genai-assignment-1/docs/design-handoff) | Application screenshots across 4 workspaces; Google Stitch design alignment; Docker Compose configuration; automated test suite verification |
| **XI. AI-Use Disclosure Appendix** | IEEE Report Appendix | Disclosure of tools used (Antigravity, Cursor, PyTorch, Optuna) for code scaffolding, debugging, and verification procedures |
