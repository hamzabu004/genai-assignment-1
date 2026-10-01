# GenAI Assignment 1

## Run the app
1. Place exported `.onnx` models in `backend/models/` (see `backend/models/download_models.sh`)
2. `cp backend/.env.example backend/.env`
3. Ensure the Oxford-IIIT Pet images are available in `research/datasets/oxford-iiit-pet/images/`; Docker Compose mounts these images and `research/val_manifest_official.json` read-only for the Universal Restoration validation picker.
4. `docker compose up --build`
5. Open http://localhost:3000

## Repo layout
- `backend/` — FastAPI + onnxruntime inference service
- `frontend/` — Next.js UI
- `research/` — training notebooks/scripts, Optuna studies (not used at runtime)
- `export/` — ONNX export + PyTorch/TF-vs-ONNX verification scripts
- `docs/` — API agreement, design notes
