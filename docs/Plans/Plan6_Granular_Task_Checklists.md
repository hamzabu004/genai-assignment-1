# Plan 6 — Granular Per-Task Checklists

This expands Plan 4 (generic simple→Optuna workflow) and Plan 5 (architecture research) into concrete, task-specific checklists. Each task has its own: data specifics, exact simple-baseline spec, exact Optuna search spec, evaluation specifics, checkpointing, and export/app-integration handoff.

---

# TASK 1 — Universal Multi-Corruption Restoration (DAE)

## 1.1 Data
- [ ] Dataloader samples one of 4 conditions per image per epoch, 25% each: clean / salt-pepper / blur / occlusion
- [ ] Salt-pepper: probability ~U(0.02, 0.15), black/white equal chance
- [ ] Blur: kernel ∈ {3,5,7}, sigma ~U(0.5, 2.5)
- [ ] Occlusion: 1–3 rectangles, 10–35% coverage, random placement
- [ ] Validation manifest generated once, stored (type, severity, params, seed) — deterministic across runs
- [ ] Test manifest with fixed severities: salt-pepper {0.03, 0.08, 0.15}; blur {(3,0.7), (5,1.5), (7,2.5)}; occlusion {~10%/1 rect, ~20%/2 rects, ~35%/3 rects}
- [ ] Confirm same 80/20 seed-42 split reused from Tasks 2 and 3 (no re-splitting)

## 1.2 Simple Baseline — Model
- [ ] Encoder: 5 conv blocks, stride-2 downsampling, channels 32→64→128→256→256, spatial 128→64→32→16→8→4
- [ ] Decoder: mirrored, upsample+conv (not transpose-conv) to avoid checkerboard artifacts
- [ ] Bottleneck: flatten 4×4×256 → FC → μ, logσ² (fixed latent dim, e.g. 128, not yet tuned)
- [ ] Reparameterization: `z = mu + eps * exp(0.5*logvar)`
- [ ] No skip connections yet in this baseline (add only after baseline is validated — see 1.6 ablation)
- [ ] GroupNorm (not BatchNorm) if batch size will be small

## 1.3 Simple Baseline — Loss & Training
- [ ] Reconstruction loss: `α·L1(x, x̂) + (1-α)·(1-SSIM(x, x̂))`, α=0.8 fixed (assignment's suggested value)
- [ ] KL term: fixed small β (e.g. 0.001–0.01), constant (no annealing yet)
- [ ] Optimizer: Adam, single fixed LR (e.g. 1e-3), single fixed batch size
- [ ] Train tiny subset (20–50 images, 1–2 epochs, CPU) — sanity check only (Plan 4, Section 1)
- [ ] Train quick full-dataset run (~10–15 epochs) on GPU — confirm loss trends down, no NaNs
- [ ] Visual check: plot ≥6 examples (clean target / corrupted input / reconstruction) across all 4 conditions

## 1.4 Transition Gate (don't proceed to Optuna until all pass)
- [ ] Loss decreases reliably across full run
- [ ] Reconstructions visually reasonable for all 4 conditions
- [ ] Checkpoint save→reload reproduces identical loss on a fixed batch
- [ ] Pipeline runs end-to-end on both Colab and University PC without crashing
- [ ] KL term is not ~0 throughout training (posterior collapse check)

## 1.5 Optuna Search
- [ ] Search space: learning rate (log-uniform), batch size, bottleneck/latent dim, encoder channel config, dropout rate, α, β
- [ ] Sampler: TPE (default); Pruner: Median or Hyperband
- [ ] Reduced epochs per trial (e.g. 15–20% of full schedule)
- [ ] n_trials sized to compute budget (target 20–40 if feasible)
- [ ] Log every trial to W&B + Optuna study object
- [ ] Checkpoint only best trial (or top-3), not every trial
- [ ] Study storage: SQLite file, committed to git after each device session (Plan 4, Section 4)

## 1.6 Post-Optuna
- [ ] Retrain winning config for full training schedule (not the reduced-epoch trial run)
- [ ] Run the skip-connection ablation now: train a with-skip variant (single skip, highest-resolution pair, partial channels only) vs the no-skip winner; compare SSIM/PSNR — this ablation *is* the assignment's required justification for any skip usage
- [ ] Decide final architecture (skip or no-skip) based on ablation result, document reasoning

## 1.7 Evaluation
- [ ] Metrics computed separately per: corruption type (clean/salt/blur/occlusion) × severity (low/med/high)
- [ ] Report PSNR, SSIM (and MSE if useful) per cell of that grid
- [ ] Generate: clean target, corrupted input, reconstruction, absolute error map — for ≥12 representative examples
- [ ] Identify and write up ≥4 meaningful failure cases (what failed, why, hypothesis)
- [ ] Posterior-collapse check result documented (pass/fail + evidence)

## 1.8 Checkpointing & Export
- [ ] Checkpoint dict includes: model_state_dict, optimizer_state_dict, epoch, val_loss, optuna_trial_number, seed, git_commit_hash
- [ ] Entry added to `checkpoints_manifest.json`
- [ ] Export to ONNX; verify PyTorch vs ONNX Runtime output within tolerance on a validation sample
- [ ] Filename convention: `task1_universal_ae.onnx`
- [ ] Confirm input/output tensor shape matches backend's `preprocessing.py` contract (128×128×3, NCHW, matching normalization)

## 1.9 App Integration Hook
- [ ] Maps to `/universal-restoration` endpoint (Plan 2, Section 3.2)
- [ ] Backend needs: corrupted_image, output_image, error_map_image, corruption_applied params, inference_time_ms

---

# TASK 2a — Corruption Classifier

## 2a.1 Data
- [ ] Same runtime corruption pipeline as Task 1, but training labels = the corruption type applied (clean/salt/blur/occlusion)
- [ ] Balanced batch sampling across the 4 classes (don't let "clean" or any single corruption dominate a batch)

## 2a.2 Simple Baseline — Model
- [ ] 4–5 conv blocks, stride-2 downsampling, channels 32→64→128→256
- [ ] Global average pooling (not flatten+large FC) to keep parameter count small
- [ ] One FC hidden layer → 4-way softmax output
- [ ] Dropout before final FC (rate fixed for baseline, tuned later)

## 2a.3 Simple Baseline — Loss & Training
- [ ] Multiclass cross-entropy loss
- [ ] Fixed LR, fixed batch size, Adam optimizer
- [ ] Tiny-subset sanity check (per Plan 4 flow) before full run
- [ ] Quick full run (~10 epochs) — confirm accuracy climbing above random baseline (25%) quickly

## 2a.4 Transition Gate
- [ ] Validation accuracy clearly above chance (>60% as an early sanity threshold, not a final target)
- [ ] No single class collapsing to ~0% recall (checks batch balancing worked)
- [ ] Checkpoint save/reload verified

## 2a.5 Optuna Search
- [ ] Search space: learning rate, batch size, conv channel configuration, dropout rate, weight decay
- [ ] Pruner enabled (kill clearly underperforming trials early)
- [ ] Log all trials to W&B

## 2a.6 Evaluation
- [ ] Overall accuracy
- [ ] Macro-averaged precision, recall, F1
- [ ] Per-class precision/recall/F1
- [ ] Normalized 4×4 confusion matrix
- [ ] Discuss any class-pair confusion (e.g. is blur confused with clean at low severity? salt-pepper vs occlusion?) — tie back to Roy et al.'s finding that some architectures handle salt-and-pepper differently than blur (Plan 5)

## 2a.7 Checkpointing & Export
- [ ] Standard checkpoint dict + manifest entry
- [ ] Export to ONNX: `task2_classifier.onnx`
- [ ] Verify predicted classes match between PyTorch and ONNX Runtime on a validation subset (not just numerical tolerance — check argmax agreement)

## 2a.8 App Integration Hook
- [ ] Feeds both `/hard-routing` (Plan 2, 3.3) and the gate initialization for Task 3
- [ ] Backend needs: class_probabilities dict, predicted_class

---

# TASK 2b — Specialist Restoration Autoencoders (salt-pepper / blur / occlusion)

## 2b.1 Data (per specialist)
- [ ] Salt-pepper specialist: trained ONLY on salt-pepper-corrupted inputs, clean targets
- [ ] Blur specialist: trained ONLY on blur-corrupted inputs, clean targets
- [ ] Occlusion specialist: trained ONLY on occlusion-corrupted inputs, clean targets
- [ ] Same clean-target images and same 80/20 split as Tasks 1 and 2a

## 2b.2 Shared Architecture Search (before training 3 separately)
- [ ] Reuse Task 1's base DAE shape (Plan 5, Task 1 architecture) as the starting point
- [ ] Run ONE shared Optuna search (on salt-pepper data, or a combined/representative subset) to find a common architecture: learning rate, bottleneck size, channel config, batch size, α (L1/SSIM weight)
- [ ] Consider a smaller latent dim than Task 1's universal model — a single-corruption specialist needs less capacity than blind multi-corruption restoration (Plan 5 rationale)
- [ ] Lock the winning architecture as the shared template for all 3 specialists

## 2b.3 Train 3 Specialists Independently
- [ ] Salt-pepper specialist: tiny-subset sanity check → full training run with shared architecture
- [ ] Blur specialist: tiny-subset sanity check → full training run
- [ ] Occlusion specialist: tiny-subset sanity check → full training run
- [ ] Each gets its own checkpoint, its own manifest entry, independently trained parameters (not shared weights)

## 2b.4 Evaluation (per specialist + combined)
- [ ] Each specialist evaluated only on its own corruption type, across low/med/high severity
- [ ] PSNR/SSIM per severity level
- [ ] Visual examples per specialist (input/output/error map)

## 2b.5 Oracle vs Predicted Routing Test
- [ ] Oracle-routing: use the deterministic test-manifest label to select which specialist processes each test image (or identity bypass if clean)
- [ ] Predicted-routing: use Task 2a's classifier output to select the specialist
- [ ] Compare oracle vs predicted routing performance — quantify the gap
- [ ] Identify and discuss specific cases where classifier misprediction caused visible restoration failure

## 2b.6 Checkpointing & Export
- [ ] 3 separate checkpoints + manifest entries
- [ ] Export each to ONNX: `task2_specialist_salt.onnx`, `task2_specialist_blur.onnx`, `task2_specialist_occlusion.onnx`
- [ ] Verify each against its PyTorch counterpart independently

## 2b.7 App Integration Hook
- [ ] Feeds `/hard-routing` (Plan 2, 3.3): selected_expert field + output_image
- [ ] Also feeds Task 3 as the expert initialization

---

# TASK 3 — Soft Mixture-of-Experts Restoration

## 3.1 Initialization (no training from scratch)
- [ ] Gate initialized from Task 2a's trained classifier weights
- [ ] Identity branch + 3 experts initialized from Task 2b's 3 trained specialists
- [ ] Confirm all 4 initializing checkpoints load correctly before starting warm-up

## 3.2 Warm-Up Stage
- [ ] Freeze all expert weights (identity branch has no learnable params to freeze, gate is the only trainable part)
- [ ] Train gate only, short schedule (e.g. a few epochs)
- [ ] Confirm gate outputs meaningful, non-degenerate weights (not collapsed to always picking one branch) before proceeding

## 3.3 Joint Fine-Tuning Stage
- [ ] Unfreeze experts
- [ ] Lower learning rate vs warm-up stage
- [ ] Joint loss: `λ1·L1 + λ2·(1-SSIM) + λ3·L_classification + λ4·L_balance`
- [ ] Initial values: λ1=0.8, λ2=0.2, λ3=0.1, λ4=0.01 (fixed for the simple baseline pass)
- [ ] Implement balance regularizer (assignment's suggested formula or your own — cite research if you deviate)
- [ ] Tiny-subset sanity check → quick full run before Optuna

## 3.4 Transition Gate
- [ ] Reconstruction quality doesn't regress vs Task 2b's oracle-routing specialists on their respective corruption types
- [ ] No expert has collapsed to near-zero average weight across the whole validation set (check before moving to Optuna, not after)
- [ ] Checkpoint save/reload verified

## 3.5 Optuna Search
- [ ] Search space: joint fine-tuning LR, temperature τ, classification weight λ3, balance weight λ4, reconstruction weighting (λ1/λ2)
- [ ] Enable trial pruning — watch specifically for routing collapse (one expert dominating all inputs) as a pruning signal, not just loss value
- [ ] Log routing-weight statistics per trial, not just final loss

## 3.6 Post-Optuna Analysis
- [ ] Average expert weights per true corruption type × severity (table or heatmap)
- [ ] Examples where one expert clearly dominates
- [ ] Examples with distributed weights across ≥2 experts
- [ ] Explicit check: any expert inactive (near-zero weight everywhere)? Any expert dominating unrelated/wrong inputs?
- [ ] Routing heatmap generated for the report

## 3.7 Checkpointing & Export
- [ ] Checkpoint captures gate + all experts as one unit (or clearly documented multi-file bundle)
- [ ] Export full pipeline to ONNX: `task3_soft_moe.onnx`
- [ ] Verify routing weights + reconstruction match PyTorch output within tolerance

## 3.8 App Integration Hook
- [ ] Feeds `/soft-mixture` (Plan 2, 3.4): routing_weights dict (4 values), dominant_expert, output_image

---

# TASK 4 — Style-Conditioned Face-to-Sketch (Conditional GAN)

## 4.1 Data
- [ ] FS2K: 2,104 paired photo-sketch, 3 style categories
- [ ] Official train/test split respected; 15% of train reserved as validation, seed 42, stratified by style
- [ ] Official test set untouched until final evaluation
- [ ] Resize photo+sketch pairs to 128×128, pairing preserved
- [ ] Any augmentation (crop/flip/rotate) applied identically to both members of a pair — verify with a visual spot-check, not just code review

## 4.2 Simple Baseline — Generator
- [ ] U-Net encoder-decoder with full skip connections (unlike Task 1, skip restriction does NOT apply here — this is image-to-image translation, not a bottleneck autoencoder)
- [ ] Style condition: learned categorical embedding for the 3 FS2K styles
- [ ] Read FS2K's own FSGAN baseline's "style-vector expansion" mechanism (Plan 5, Task 4) before finalizing exactly where/how the embedding is injected into the generator
- [ ] Fixed, modest embedding dimension for the baseline (tune later via Optuna)

## 4.3 Simple Baseline — Discriminator
- [ ] PatchGAN discriminator, judges local patch realism, not whole-image realism
- [ ] Receives photograph + style condition + (real or generated) sketch

## 4.4 Simple Baseline — Loss & Training
- [ ] Generator loss: adversarial + λ_L1 · L1(target, generated), λ_L1 = 100 (assignment's suggested starting value)
- [ ] Discriminator/adversarial loss: BCE-with-logits
- [ ] Fixed G/D learning rates, fixed batch size, small fixed base channel count for baseline
- [ ] Tiny-subset sanity check (a handful of pairs, 1–2 epochs) before any real run
- [ ] Quick run on full data — confirm D real/fake losses and G adversarial/reconstruction losses are all logging correctly and moving in sensible directions (D not immediately saturating to 0/1)

## 4.5 Transition Gate
- [ ] Generated sketches are visually sketch-like (even if rough) — not noise, not a copy of the input photo
- [ ] D-real and D-fake losses both non-degenerate (D hasn't collapsed to trivially perfect or trivially fooled)
- [ ] Style conditioning has a visible effect: same photo + different style token produces visibly different output
- [ ] Checkpoint save/reload verified

## 4.6 Optuna Search
- [ ] Reduced-epoch trials (GAN training is expensive — Optuna trials use fewer epochs than final training, per assignment's allowance)
- [ ] Search space: G learning rate, D learning rate, batch size, base channel count, dropout rate, style-embedding dimension, λ_L1
- [ ] Log D-real, D-fake, G-adversarial, G-reconstruction losses separately per trial, not just a single combined metric

## 4.7 Post-Optuna
- [ ] Retrain winning config for the FULL training schedule (not the reduced-epoch trial run)
- [ ] Log fixed-validation-image samples at regular intervals across the full run to show generator progression over time

## 4.8 Evaluation
- [ ] Qualitative: generated sketch vs ground-truth sketch, side by side, across all 3 styles
- [ ] Show same input photo across all 3 styles to demonstrate conditioning works
- [ ] Discuss failure cases (style bleeding, missing facial detail, artifacts)
- [ ] Separate loss curves reported: D-real, D-fake, G-adversarial, G-reconstruction, validation metric over training

## 4.9 Checkpointing & Export
- [ ] Checkpoint generator and discriminator separately (only generator is needed at inference)
- [ ] Manifest entry for both, but only generator proceeds to export
- [ ] Export generator only to ONNX: `task4_generator.onnx`
- [ ] Verify ONNX generator output matches PyTorch generator output (visual + numerical) across all 3 styles

## 4.10 App Integration Hook
- [ ] Feeds `/face-to-sketch` (Plan 2, 3.5): sketch_image, style_used, inference_time_ms
- [ ] Confirm style string mapping (`style_1`/`style_2`/`style_3`) matches the same index order used during training — a silent off-by-one here would produce a working-looking but wrong-style app

---

# Cross-Task Final Checklist
- [ ] All 4 tasks' final models retrained to full schedule (not left as reduced-epoch Optuna winners)
- [ ] All exports verified against their PyTorch/TF source within stated tolerance
- [ ] All checkpoints portable-tested (loaded fresh on a different device than they were trained on)
- [ ] `checkpoints_manifest.json` complete and accurate for every model
- [ ] Every task's Optuna study fully documented (search space, trial count, best config, importance plot) for the report
