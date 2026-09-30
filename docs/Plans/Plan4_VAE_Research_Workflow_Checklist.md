# Plan 4 — VAE Research Workflow: Simple Baseline → Optuna, Multi-Device

**Scope assumption (from your answer):** VAE applies wherever the assignment specifies "autoencoder" — i.e. **Task 1 (universal restoration)** and **Task 2's three specialists (salt-pepper, blur, occlusion)**. Task 3 reuses these trained experts + a gate (no new autoencoder to build), Task 4 is a GAN (not in scope here).

**Sync assumption:** GitHub push/pull for code (Google Drive ruled out — mounting blocked on University PC). Checkpoints handled separately from code (see Section 5) since large binaries don't belong in normal git history.

**Important — report requirement:** The assignment's baseline loss (L1 + SSIM) doesn't include a KL term. Using a VAE instead of a plain autoencoder is an architecture decision you must justify with research in the report (why VAE > plain AE for this restoration task, what alternatives you considered, e.g. plain deterministic AE, denoising AE, VAE, VQ-VAE). Keep notes on this as you go — don't leave it for write-up time.

---

## 0. One-Time Environment & Repo Setup

- [ ] Create GitHub repo (or `research/` folder in existing monorepo) with structure:
  ```
  research/
    src/                    # shared, importable code (NOT duplicated per-notebook)
      datasets.py
      corruptions.py
      models_vae.py
      losses.py
      train_loop.py
      checkpoint_utils.py
    notebooks/
      task1_vae_simple.ipynb
      task1_vae_optuna.ipynb
      task2_classifier.ipynb
      task2_specialist_salt_simple.ipynb
      task2_specialist_salt_optuna.ipynb
      ... (repeat pattern per specialist)
    configs/
      task1_best_config.yaml
      task2_salt_best_config.yaml
      ...
    checkpoints_manifest.json   # tracked in git, small text file (see Section 5)
  ```
- [ ] `requirements.txt` or `environment.yml` pinned with exact versions (torch, torchvision, optuna, wandb, scikit-image for SSIM, etc.)
- [ ] `.gitignore` includes: `*.pt`, `*.pth`, `*.ckpt`, `wandb/`, `outputs/`, `.ipynb_checkpoints/`, Optuna `.db` files (unless small enough — decide in Section 5)
- [ ] Install `nbstripout` (or similar) and enable it on the repo — strips notebook output before commit so diffs stay small and image blobs don't bloat git history across 3 devices
- [ ] Set up Weights & Biases (W&B) account — chosen over local MLflow because it's cloud-hosted, so all 3 devices log to the same dashboard without needing file-sync (just an API key + internet)
- [ ] Confirm `wandb login` works from: laptop, Colab, University PC — test **now**, not mid-training (university networks sometimes block external services)
- [ ] Test **Git LFS feasibility on the University PC specifically** — since it's your own persistent account, check if you can install the `git-lfs` binary without admin rights (it's a single portable executable). Do this test early; if blocked, fall back to Hugging Face Hub model repo (also git-based, free, no admin rights needed, works over plain `git push`) — decide fallback now, don't discover the blocker mid-project

---

## 1. Dev → Validate → Run Flow (across devices)

- [ ] **Write & validate on laptop first** (or whichever device you write code on), using a **tiny subset** (~20–50 images) and **1–2 epochs**, CPU is fine — goal is only to catch bugs (shape mismatches, NaN losses, logging errors), not to produce a real model
- [ ] Confirm: forward pass runs, loss computes without NaN/Inf, checkpoint save+load round-trips correctly, W&B logs a test run successfully
- [ ] Commit + push to GitHub once the tiny-subset run is clean
- [ ] Pull on Colab → re-run the same tiny-subset sanity check (catches GPU-specific issues: mixed precision, CUDA errors, batch-size-driven OOM) **before** launching any real training
- [ ] Pull on University PC → repeat the same tiny-subset sanity check
- [ ] Only after sanity checks pass on a device, launch the actual (larger) training run there
- [ ] Add a "device check" cell at the top of every notebook: prints `torch.cuda.is_available()`, GPU name, available VRAM — confirms environment before a long run starts
- [ ] Tag every W&B run with a `device` field (`laptop` / `colab` / `uni-pc`) so you can trace which environment produced which result later

---

## 2. Phase 1 — Simple VAE Baseline

### Include
- [ ] Fixed architecture: conv encoder (stride-downsampling) → `mu`, `logvar` heads → reparameterization (`z = mu + eps * std`) → conv/transpose-conv decoder
- [ ] Fixed bottleneck/latent dim (pick one reasonable value, e.g. 128 — don't search yet)
- [ ] Reconstruction loss = assignment's required L1 + SSIM combination, α = 0.8 (assignment's suggested starting value, unchanged for now)
- [ ] KL divergence term added with a **small, fixed** β (e.g. β = 0.001–0.01) so it regularizes without dominating reconstruction — the goal here is restoration quality, not generative sampling
- [ ] Single optimizer (Adam), single fixed learning rate, single fixed batch size
- [ ] Constant or simple StepLR schedule — nothing elaborate
- [ ] Train on tiny subset first (Section 1), then a quick full-dataset run (~10–15 epochs) to confirm loss trends down and reconstructions look sane
- [ ] Manual checkpoint saved at end of run + every few epochs
- [ ] Plot a handful of reconstructions (clean/corrupted/output) to visually sanity-check
- [ ] Minimal W&B logging — just enough to confirm the logging pipeline itself works end-to-end

### Explicitly skip in this phase
- [ ] No Optuna, no hyperparameter sweeps of any kind
- [ ] No architecture search (channel counts, depth fixed)
- [ ] No β-annealing schedule (keep β constant for now)
- [ ] No SSIM-weight (α) tuning — use the assignment's suggested 0.8 as-is
- [ ] No fancy LR schedulers, no warm restarts
- [ ] No multi-GPU / distributed training
- [ ] No large-batch memory tuning — pick a batch size that comfortably fits on the weakest device (laptop/CPU test), adjust up only per-device if needed
- [ ] No elaborate resumable-training machinery yet — just "save latest + save best," nothing more sophisticated

### Transition criteria (don't move to Optuna until all true)
- [ ] Loss curves decrease reliably across a full run, no NaNs
- [ ] Reconstructions visually reasonable across all 4 input conditions (clean/salt/blur/occlusion)
- [ ] Full pipeline runs end-to-end without crashing on **both** Colab and University PC
- [ ] Checkpoint save → reload → produces identical loss on the same batch (explicitly test this, don't assume it works)
- [ ] W&B logging confirmed working consistently from all 3 devices
- [ ] Code pushes/pulls cleanly via GitHub with no merge conflicts; `.gitignore` correctly keeps checkpoints and notebook outputs out of git

---

## 3. Phase 2 — Optuna Search

### Include
- [ ] Search space (per model — universal AE and each of the 3 specialists get their own study):
  - [ ] Learning rate (log-uniform)
  - [ ] Batch size (categorical, device-memory-aware)
  - [ ] Bottleneck/latent dimension
  - [ ] Encoder channel configuration
  - [ ] Dropout rate
  - [ ] α (L1 vs SSIM weight) — assignment requires this be tuned, not just assumed at 0.8
  - [ ] β (KL weight) — extra hyperparameter introduced by your VAE choice, tune it since posterior collapse or over-regularization risk depends heavily on it
- [ ] Use Optuna's default TPE sampler (efficient for this kind of mixed continuous/categorical space) — don't hand-roll grid search
- [ ] Use a pruner (Median or Hyperband) to kill clearly bad trials early — important given limited/interruptible compute (Colab sessions especially)
- [ ] Run each trial with **reduced epochs** relative to full training (Optuna trials should be cheap; you retrain the winner fully afterward — see below)
- [ ] Log every trial (params + intermediate/final metric) to W&B, in addition to Optuna's own study object
- [ ] Save checkpoints for the **best trial only** (or top-k, e.g. top-3) — don't checkpoint every trial, it'll blow up storage fast
- [ ] After the study completes: retrain the winning config for the **full training schedule** (not just the reduced-epoch trial run) — this final model is what gets exported to ONNX later
- [ ] Record and include in the report: full search space definition, number of completed trials, best trial's params + value, an Optuna parameter-importance plot (`optuna.visualization.plot_param_importances`), and the optimization-history plot
- [ ] After training, explicitly check for **posterior collapse**: confirm the KL term isn't ~0 throughout training (which would mean the decoder is ignoring the latent and it's degenerated into a near-deterministic AE) — note this check and its result in the report regardless of outcome

### Explicitly avoid
- [ ] Don't tune everything at once on the first pass — start with the core list above; adding more dimensions multiplies the trials you need for a meaningful search
- [ ] Don't run full-epoch training inside every trial — wastes compute that pruning + reduced-epoch trials would save
- [ ] Don't discard pruned-trial data — keep the study storage even for pruned trials, useful for the report's search-space discussion
- [ ] Don't hardcode a device inside the notebook (e.g. `.cuda()` unconditionally) — use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` so the same notebook runs unmodified on laptop (CPU sanity check) and GPU devices
- [ ] Don't run the **same** Optuna study concurrently from two devices pointed at different local storage — you'll get two divergent, unmergeable studies. Either run one device at a time (recommended, matches your sequential dev→cloud→uni flow) or use distinct study names per device and treat results separately if you ever do run in parallel
- [ ] Don't expect exact reproducibility across devices (different GPUs can give slightly different results even with fixed seeds) — fix seeds anyway for as much reproducibility as possible, but note this as a stated limitation in the report rather than chasing perfect determinism

---

## 4. Optuna Study Persistence Across Devices (sequential workflow)

Since you're moving between devices rather than running in parallel, keep the Optuna study itself portable:

- [ ] Use Optuna's SQLite storage backend (`sqlite:///task1_study.db`) — a single small file per study
- [ ] Because this file is small, decide explicitly whether it lives in git (small text-like binary, generally fine to commit) or is transferred alongside checkpoints via Git LFS/HF Hub — pick one and be consistent (recommendation: commit it directly if it stays under a few MB, simplest option given your git-only sync constraint)
- [ ] Before switching devices mid-study: commit/push the latest `.db` file
- [ ] On the new device: pull, then resume the study with the same `study_name` and `storage` path — Optuna will continue from existing trials rather than restarting
- [ ] Verify resumption worked: check `len(study.trials)` matches expectations before launching new trials

---

## 5. Checkpoint Management

- [ ] Checkpoint dict contents (every save): `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_loss`, `optuna_trial_number` (if applicable, else `None`), `random_seed`, `git_commit_hash` (ties checkpoint back to the exact code version that produced it)
- [ ] Save two files per run: `latest.pt` (always overwritten) and `best.pt` (only overwritten when val loss improves) — avoids losing progress if a later epoch degrades
- [ ] **Never commit `.pt`/`.pth` files directly into normal git history** — use Git LFS (if the earlier feasibility test on University PC passed) or push to a Hugging Face Hub model repo as the agreed fallback
- [ ] Maintain `checkpoints_manifest.json` in the main git repo (plain text, git-friendly, always synced) logging every checkpoint: filename, task, phase (`simple` / `optuna_trial_N` / `final`), val loss, epoch, device trained on, timestamp, git commit hash — this survives normally in git even though the actual weight files live elsewhere
- [ ] When resuming training on a **different** device: pull latest git commit → fetch the referenced checkpoint from LFS/HF Hub → **verify it loads and reproduces the last logged val loss on a fixed batch** before continuing — catches silent corruption from the transfer
- [ ] Auto-save a checkpoint periodically during long runs (e.g. every N minutes or every epoch), not just at the very end — Colab can disconnect on idle/timeout mid-run, and you want to resume rather than restart

---

## 6. Notebook Hygiene

- [ ] One notebook per task + phase (e.g. `task1_vae_simple.ipynb`, `task1_vae_optuna.ipynb`) rather than one giant notebook — keeps runs isolated and diffs manageable
- [ ] Notebooks stay thin: import from `src/` modules (dataset class, corruption functions, model definitions, loss functions, training loop) rather than redefining the same code inline in every notebook — this is what actually needs to work identically across devices, so keeping it in `.py` files (not notebook cells) makes it easier to test and sync via git
- [ ] Standard notebook cell order: config/imports → device check → data loading → model definition (imported) → training function call (simple run **or** Optuna objective) → evaluation/visualization
- [ ] Set random seeds at the top of every notebook
- [ ] `nbstripout` active (Section 0) so outputs don't get committed and bloat the repo across 3 devices

---

## 7. Cross-Device Practical Checks (do these once, early)

- [ ] Python version and key package versions (torch, optuna, wandb, scikit-image) match — or are pinned identically via `requirements.txt` — on laptop, Colab, and University PC
- [ ] Git LFS install feasibility confirmed on University PC (Section 0) — fallback (Hugging Face Hub) ready if blocked
- [ ] University PC network allows W&B and (if used) Hugging Face Hub access — test before relying on it for a real run
- [ ] Compare GPU/VRAM available on Colab vs University PC — set device-appropriate batch sizes (a small `configs/device_overrides.yaml` mapping device → batch size is a clean way to avoid editing notebook code per device)
- [ ] Plan around Colab's session time limits and idle disconnects — confirm your checkpoint-resume logic (Section 5) actually handles an interrupted run gracefully, test this deliberately (kill the runtime mid-training on purpose once, then resume)

---

## 8. Final Verification (per model, before moving to export)

- [ ] Final config was retrained to the **full** schedule, not left as the reduced-epoch Optuna trial run
- [ ] Best checkpoint is loadable from a **fresh** session/device (test actual portability, don't assume from context)
- [ ] Posterior-collapse check documented (KL term behavior over training) — include the result either way in the report
- [ ] Checkpoint's model class version matches what the eventual ONNX export script expects (state_dict keys line up) — worth a quick load-test against the export script stub before considering the model "done"
