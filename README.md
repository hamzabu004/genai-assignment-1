# GenAI Assignment 1

This guide covers exporting the trained ONNX models, moving them to the backend machine, and starting the backend and frontend from the command line.

## 1. Export the models

On the machine that has the `research/checkpoints/*final_best.pt` files, run these from the repository root:

```bash
source .venv/bin/activate
python export/export_all_onnx.py
```

The exporter creates and verifies all seven models in `backend/models/` and writes `export/reports/final_best_onnx_validation.json`. If this is a new Python environment, install the research requirements first:

```bash
python -m pip install -r research/requirements.txt
```

The ONNX files are git-ignored, so transfer them separately or use W&B as described below.

## 2. Transfer models to the backend machine

### Using W&B Artifacts

Set `WANDB_API_KEY` and, if needed, `WANDB_ENTITY` and `WANDB_PROJECT` in the shell or repository `.env` on the export machine. Then upload the verified bundle:

```bash
python export/upload_onnx_wandb.py
```

On the backend machine, from the repository root, set the same W&B variables and fetch the latest bundle:

```bash
python -m pip install wandb
python backend/models/fetch_wandb_models.py
```

To fetch a specific version instead, pass its artifact path:

```bash
python backend/models/fetch_wandb_models.py --artifact ENTITY/PROJECT/final-best-onnx:v0
```

Fetching verifies the file hashes from the export report and places the seven ONNX files in `backend/models/`.

### Transfer files directly

Alternatively, copy all seven `.onnx` files from `backend/models/` on the export machine to the same directory in the backend checkout. Keep the export report alongside them if you want the verification metadata available there.

## 3. Start backend and frontend with Docker Compose

From the repository root on the machine that will run the application, with Docker Compose installed and models present in `backend/models/`:

```bash
docker compose up --build
```

This starts the backend at <http://localhost:8000> and the frontend at <http://localhost:3000>. Stop both with `Ctrl+C`, or run `docker compose down` from another terminal.

The universal-restoration validation picker also uses `research/val_manifest_official.json` and the images in `research/datasets/oxford-iiit-pet/images/`; make sure these are available before starting Compose.

## 4. Start backend and frontend directly

Open two terminals at the repository root. Create the backend environment once and install its dependencies:

```bash
python -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
```

In terminal 1, start the API:

```bash
cd backend
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In terminal 2, install frontend dependencies and start Next.js:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. The frontend defaults to the API at <http://localhost:8000>. For a frontend running on a different machine, set `NEXT_PUBLIC_API_BASE_URL` to the backend's reachable URL before starting Next.js, and configure the backend's `ALLOWED_ORIGINS` to the frontend's origin.

## Useful commands

```bash
# Export and verify the final-best ONNX files
python export/export_all_onnx.py

# Upload and fetch the W&B model bundle
python export/upload_onnx_wandb.py
python backend/models/fetch_wandb_models.py

# Start or stop both app services with Docker Compose
docker compose up --build
docker compose down
```

## Repository layout

- `backend/` — FastAPI API, ONNX Runtime models, and W&B fetch script
- `frontend/` — Next.js interface
- `research/` — training notebooks, checkpoints, and research code
- `export/` — ONNX exporter, W&B uploader, and verification reports
- `docs/` — API and design documentation
