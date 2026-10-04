# Task 2b Specialist Architecture and Optuna Search

This report describes the three Task 2b corruption specialists (salt and pepper, blur, and occlusion), their shared architecture, and the Optuna search used to choose their common hyperparameters. It distinguishes the architecture used by the recorded search from the current architecture in the working tree.

## Specialist setup

Task 2b trains three separate `ConvDAE` restoration models. Each model sees examples of its own corruption type and reconstructs the matching clean image. The same selected architecture and hyperparameters are shared across all three models; their learned weights are trained independently.

The specialist objective in the executed notebook is the weighted reconstruction loss:

\[
\mathcal{L}=\alpha\,\mathcal{L}_{L1}+(1-\alpha)(1-\mathrm{SSIM})
\]

## Architecture used in the recorded search

The saved Optuna result was produced with the earlier `ConvDAE` shape: five encoder stages downsampled a 128×128 input to a 4×4 feature map. The encoder channel widths are defined by `base_channels` as `[b, 2b, 4b, 8b, 8b]`. For the winning `b = 64`, this is `[64, 128, 256, 512, 512]`.

| Encoder stage | Input → output spatial size | Channels (winning width) | Main operations |
|---|---:|---:|---|
| enc1 | 128×128 → 64×64 | 3 → 64 | 4×4 stride-2 convolution, GroupNorm, LeakyReLU, dropout |
| enc2 | 64×64 → 32×32 | 64 → 128 | 4×4 stride-2 convolution, GroupNorm, LeakyReLU, dropout |
| enc3 | 32×32 → 16×16 | 128 → 256 | 4×4 stride-2 convolution, GroupNorm, LeakyReLU |
| enc4 | 16×16 → 8×8 | 256 → 512 | 4×4 stride-2 convolution, GroupNorm, LeakyReLU |
| enc5 | 8×8 → 4×4 | 512 → 512 | 4×4 stride-2 convolution, GroupNorm, LeakyReLU |

The 4×4×512 tensor is flattened and projected by a fully connected layer to the selected `latent_dim` (256 for the recorded winner). The decoder maps the latent vector back to 4×4×512, then upsamples through 8×8, 16×16, 32×32, 64×64, and 128×128. Each decoder stage uses bilinear upsampling and a 3×3 convolution with normalization and activation; the final stage adds an RGB output convolution and sigmoid. The Task 2b notebook disables the optional high-resolution skip connection.

## Current working-tree architecture

The current `research/src/models_dae.py` has since been changed to use six encoder downsampling stages, reaching a **2×2** bottleneck. It adds `enc6` for 4×4 → 2×2 and a matching `dec6` for 2×2 → 4×4. Therefore there are six upsampling stages after the latent vector is expanded back to the 2×2 feature map, followed by the final RGB output convolution. With `base_channels = 64`, the added 2×2 stage remains at 512 channels.

The recorded Optuna winner and validation loss were obtained before this architecture change. They are historical results and have not been re-validated for the new 2×2 architecture. The current search notebook still describes the former search space and should be rerun if results for the 2×2 model are needed.

## Optuna study and search space

The executed notebook (`research/notebooks/task2b_specialists_optuna.ipynb`) conducts a shared search on the salt-and-pepper training/validation corruption split. It minimizes the best validation composite reconstruction loss across 15 epochs per trial, with 20 trials, then applies the winning settings while training each specialist. The sampler/pruner are configured in the notebook; pruning uses Optuna's median pruner.

| Hyperparameter | Search space | Sampling |
|---|---|---|
| Learning rate (`lr`) | 1×10⁻⁴ to 1×10⁻³ | Continuous, log scale |
| Batch size (`batch_size`) | 16, 32, 64 | Categorical |
| Latent dimension (`latent_dim`) | 64, 128, 256 | Categorical |
| Base channels (`base_channels`) | 32, 64 | Categorical |
| Dropout (`dropout`) | 0.0 to 0.3 | Continuous, linear |
| L1/SSIM weight (`alpha`) | 0.5 to 0.95 | Continuous, linear |

`beta` does **not** appear in the executed Task 2b DAE objective or its search space. It appears in a stale Task 2b notebook generator template, which imports a VAE that is not the implementation used by the executed notebook. This report follows the executed notebook and the saved Task 2b results.

## Recorded winner

The three specialist config files currently store the same shared winning result:

| Parameter | Selected value |
|---|---:|
| `lr` | 0.0001067657 |
| `batch_size` | 32 |
| `latent_dim` | 256 |
| `base_channels` | 64 |
| `dropout` | 0.0260257 |
| `alpha` | 0.9471441 |
| Recorded best validation loss | 0.1086992 |

These values are stored in `research/configs/task2b_{salt,blur,occlusion}_best_config.yaml`. The config files copy one shared search result; they do not represent three independent Optuna studies.

## Source files

- Architecture: `research/src/models_dae.py`
- Executed search and retraining notebook: `research/notebooks/task2b_specialists_optuna.ipynb`
- Simple specialist training notebook: `research/notebooks/task2b_specialists_simple.ipynb`
- Shared winning configs: `research/configs/task2b_salt_best_config.yaml`, `research/configs/task2b_blur_best_config.yaml`, and `research/configs/task2b_occlusion_best_config.yaml`
- Notebook generator: `research/generate_notebooks.py` (contains a stale Task 2b Optuna template that differs from the executed notebook)
