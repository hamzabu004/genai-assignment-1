# Plan 2 — Coding Agent Plan: Folder Structure & API Agreement

Give this file directly to the coding agent alongside Plan 3's scaffold output. It defines the contract between frontend and backend so both can be built/tested independently.

---

## 1. Repository Structure (monorepo)

```
genai-assignment/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entrypoint, mounts routers, CORS
│   │   ├── core/
│   │   │   ├── config.py            # env vars, model paths, constants (IMG_SIZE=128, etc.)
│   │   │   └── logging.py
│   │   ├── routers/
│   │   │   ├── health.py
│   │   │   ├── universal_restoration.py
│   │   │   ├── hard_routing.py
│   │   │   ├── soft_mixture.py
│   │   │   └── face_to_sketch.py
│   │   ├── schemas/                 # Pydantic request/response models (mirrors Section 3 below)
│   │   │   ├── universal_restoration.py
│   │   │   ├── hard_routing.py
│   │   │   ├── soft_mixture.py
│   │   │   └── face_to_sketch.py
│   │   ├── services/
│   │   │   ├── onnx_runtime_manager.py   # loads & caches all ONNX sessions at startup
│   │   │   ├── preprocessing.py          # SHARED preprocessing, must match research code exactly
│   │   │   ├── postprocessing.py
│   │   │   └── corruption.py             # runtime corruption application (matches training defs)
│   │   └── utils/
│   │       └── timing.py
│   ├── models/                      # .onnx files live here (or downloaded via script)
│   │   └── download_models.sh
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/
│   ├── app/                         # Next.js App Router
│   │   ├── layout.tsx               # sidebar + shared shell
│   │   ├── page.tsx                 # redirect to /universal-restoration
│   │   ├── universal-restoration/page.tsx
│   │   ├── hard-routing/page.tsx
│   │   ├── soft-mixture/page.tsx
│   │   └── face-to-sketch/page.tsx
│   ├── components/
│   │   ├── layout/Sidebar.tsx
│   │   ├── ui/Button.tsx            # handles disabled + loading states
│   │   ├── ui/Loader.tsx
│   │   ├── ui/UploadZone.tsx
│   │   ├── ui/ImagePanel.tsx
│   │   ├── ui/BarChart.tsx          # classifier probs / routing weights
│   │   ├── ui/Pill.tsx
│   │   └── ui/ErrorBanner.tsx
│   ├── lib/
│   │   ├── api.ts                   # typed fetch wrappers, one per endpoint
│   │   └── types.ts                 # TS types mirroring backend Pydantic schemas
│   ├── hooks/
│   │   └── useInference.ts          # shared loading/error/result state logic
│   ├── public/
│   ├── next.config.js
│   ├── tailwind.config.ts           # borderRadius: 0 globally, Apple font stack
│   ├── package.json
│   └── Dockerfile
│
├── research/                        # training notebooks/scripts (not used by backend at runtime)
├── export/                          # ONNX export + verification scripts
├── docs/
│   └── api-agreement.md             # this section 3, kept in sync
├── docker-compose.yml
└── README.md
```

---

## 2. Environment Variables

**backend/.env.example**
```
MODEL_DIR=./models
IMG_SIZE=128
ALLOWED_ORIGINS=http://localhost:3000
LOG_LEVEL=info
```

**frontend/.env.example**
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## 3. API Agreement

Base URL: `http://localhost:8000`
All image-upload endpoints accept `multipart/form-data`. All responses are `application/json` with images returned as base64-encoded PNG strings (`data:image/png;base64,...`) to keep frontend integration simple (no separate file-serving endpoint needed).

Standard error response shape (all endpoints):
```json
{
  "error": true,
  "message": "human-readable message",
  "detail": "optional technical detail"
}
```
HTTP status codes: `400` invalid input, `422` validation error, `500` inference failure.

### 3.1 `GET /health`
Response `200`:
```json
{ "status": "ok", "models_loaded": ["universal_ae", "classifier", "specialist_salt", "specialist_blur", "specialist_occlusion", "soft_moe", "generator"] }
```

### 3.2 `POST /universal-restoration`
Request (`multipart/form-data`):
| field | type | notes |
|---|---|---|
| `image` | file | required |
| `corruption_type` | string | one of `clean, salt_pepper, blur, occlusion` — optional; if omitted, image used as-is |
| `severity` | string | one of `low, medium, high` — required if corruption_type != clean |

Response `200`:
```json
{
  "input_image": "data:image/png;base64,...",
  "corrupted_image": "data:image/png;base64,...",
  "output_image": "data:image/png;base64,...",
  "error_map_image": "data:image/png;base64,...",
  "corruption_applied": { "type": "blur", "severity": "medium", "params": {"kernel": 5, "sigma": 1.5} },
  "inference_time_ms": 42.3
}
```

### 3.3 `POST /hard-routing`
Request: same as 3.2 (`image`, optional `corruption_type`, `severity`)

Response `200`:
```json
{
  "input_image": "data:image/png;base64,...",
  "corrupted_image": "data:image/png;base64,...",
  "output_image": "data:image/png;base64,...",
  "class_probabilities": { "clean": 0.02, "salt_pepper": 0.05, "blur": 0.88, "occlusion": 0.05 },
  "predicted_class": "blur",
  "selected_expert": "blur_specialist",
  "inference_time_ms": 37.1
}
```

### 3.4 `POST /soft-mixture`
Request: same as 3.2

Response `200`:
```json
{
  "input_image": "data:image/png;base64,...",
  "corrupted_image": "data:image/png;base64,...",
  "output_image": "data:image/png;base64,...",
  "routing_weights": { "identity": 0.03, "salt_pepper": 0.05, "blur": 0.82, "occlusion": 0.10 },
  "dominant_expert": "blur",
  "inference_time_ms": 51.4
}
```

### 3.5 `POST /face-to-sketch`
Request (`multipart/form-data`):
| field | type | notes |
|---|---|---|
| `image` | file | required, facial photo |
| `style` | string | one of `style_1, style_2, style_3` — required |

Response `200`:
```json
{
  "input_image": "data:image/png;base64,...",
  "sketch_image": "data:image/png;base64,...",
  "style_used": "style_2",
  "inference_time_ms": 63.0
}
```

---

## 4. Preprocessing Contract (must match `research/` exactly — single source of truth)
Defined once in `backend/app/core/config.py` and referenced by both `services/preprocessing.py` and the research code (copy the constants file into `research/` too, don't hand-duplicate values):
```python
IMG_SIZE = 128
NORMALIZE_MEAN = [0.5, 0.5, 0.5]   # confirm against actual training config before finalizing
NORMALIZE_STD  = [0.5, 0.5, 0.5]
CHANNEL_ORDER  = "RGB"
TENSOR_LAYOUT  = "NCHW"
```
Backend must NOT hardcode these separately in more than one place — import from `core/config.py` everywhere.

---

## 5. Testing Contract
- Backend: `pytest` hits each router with a sample image fixture, asserts response schema + status 200
- Frontend: at minimum, manual QA checklist per workspace (upload → loading state → result render → error state)
- Cross-check: for each endpoint, compare backend JSON output image against the ONNX verification script's output image (Plan 3 / export stage) using the same sample input
