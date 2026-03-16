# Medical SAM3 — Web Demo

A locally-running web interface for Medical SAM3 lesion detection on ultrasound
(and other medical) images. Upload an image, enter a text prompt (e.g. "breast
tumor"), and get back a segmentation overlay with detected lesion masks and
confidence scores.

## Prerequisites

- macOS (Apple Silicon or Intel) — uses MPS when available, falls back to CPU
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- ~15 GB free disk space (checkpoint ≈ 10 GB)
- HuggingFace account (for gated model access — run `huggingface-cli login`)

## Quick Start

```bash
# 1. Install everything (conda env, SAM3, Medical-SAM3, deps)
bash install.sh

# 2. Activate the environment
conda activate medsam3

# 3. Download the Medical-SAM3 checkpoint (~10 GB)
python download_ckpt.py

# 4. Start the server
python backend/app.py
```

Open **http://localhost:8000** in your browser.

## Usage

1. Upload an ultrasound (or other medical) image via drag-and-drop or file picker.
2. Pick a suggested prompt or type a custom one (e.g. "breast tumor", "liver lesion").
3. Adjust the confidence threshold slider (lower = more sensitive).
4. Click **Segment** — the overlay and detection cards appear on the right.

## Project Structure

```
MedicalSAM3/
├── install.sh              # one-shot environment setup
├── download_ckpt.py        # checkpoint downloader
├── backend/
│   ├── app.py              # FastAPI server (run this)
│   └── inference.py        # model loading & predict()
├── frontend/
│   └── index.html          # self-contained UI
├── checkpoints/
│   └── checkpoint.pt       # (downloaded separately)
├── third_party/
│   ├── sam3/               # facebook/sam3 (cloned by install.sh)
│   └── Medical-SAM3/       # AIM-Research-Lab/Medical-SAM3
└── README.md
```

## API Endpoints

| Method | Path       | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Serves the web UI                  |
| GET    | `/health`  | Model status and device info       |
| GET    | `/prompts` | List of suggested text prompts     |
| POST   | `/predict` | Run segmentation (multipart form)  |

### POST /predict

**Form fields:**

| Field     | Type   | Default         | Description                |
|-----------|--------|-----------------|----------------------------|
| image     | file   | (required)      | Image file (PNG, JPEG, …)  |
| prompt    | string | "breast tumor"  | Text prompt                |
| threshold | float  | 0.1             | Confidence threshold 0–1   |

**Response (JSON):**

```json
{
  "ok": true,
  "num_detections": 1,
  "scores": [0.87],
  "boxes": [[120, 45, 310, 280]],
  "overlay_b64": "<base64 PNG>",
  "elapsed_ms": 2340.5,
  "image_size": [512, 512]
}
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sam3` | Re-run `pip install -e third_party/sam3` inside the conda env |
| `Cannot import sam3_inference` | Ensure `third_party/Medical-SAM3` is cloned and contains `inference/sam3_inference.py` |
| Download fails (401/403) | Run `huggingface-cli login` — the checkpoint repo may be gated |
| MPS errors on Apple Silicon | Already handled: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically |
| Out of memory | The model is 10 GB; ensure you have ≥16 GB RAM. Try `MEDSAM3_DEVICE=cpu` |
| Port in use | `MEDSAM3_PORT=9000 python backend/app.py` |

## Environment Variables

| Variable                      | Default | Description                   |
|-------------------------------|---------|-------------------------------|
| `MEDSAM3_PORT`                | 8000    | Server port                   |
| `PYTORCH_ENABLE_MPS_FALLBACK` | 1       | Auto-set; allows CPU fallback |

## Credits

- [Medical-SAM3](https://github.com/AIM-Research-Lab/Medical-SAM3) — AIM Research Lab
- [SAM3](https://github.com/facebookresearch/sam3) — Meta FAIR
- Checkpoint: [ChongCong/Medical-SAM3](https://huggingface.co/ChongCong/Medical-SAM3)
