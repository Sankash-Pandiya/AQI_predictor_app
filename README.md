# AQI Predictor — MLOps Pipeline

Predicts next-day **Air Quality Index** for Indian cities using a scikit-learn
`HistGradientBoostingRegressor` pipeline served via **FastAPI**, containerised
with **Docker**, and deployed automatically through **GitHub Actions → GHCR**.

---

## Project structure

```
AQI_predictor/
├── app/
│   ├── __init__.py
│   ├── predictor.py          # FastAPI app + inference logic
│   └── transformers.py       # Winsorizer & custom transformers
├── tests/
│   ├── __init__.py
│   └── test_predictor.py     # pytest unit + integration tests
├── .github/
│   └── workflows/
│       └── ci_cd.yml         # GitHub Actions CI/CD pipeline
├── dashboard.py              # Streamlit Interactive Dashboard
├── aqi_pipeline.pkl          # Trained sklearn pipeline (joblib)
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Local dev (API + Dashboard)
├── requirements.txt          # Runtime dependencies
├── pyproject.toml            # Ruff + pytest config
├── .dockerignore
├── .gitignore
└── venv/                     # Local Python virtual environment (not committed)
```

---

## Quickstart (local)

### 1 — Activate virtual environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 2 — Run the Streamlit Dashboard (Showcase)

```bash
streamlit run dashboard.py
```

Then open <http://localhost:8501> to interact with the dashboard:
- 🔮 **Real-Time Forecasting**: Dynamic gauge, safety categories, and health advisories
- 🕹️ **Presets**: Instant testing for Delhi Smog, Bengaluru Urban, Chennai Breeze, etc.
- 🧪 **Pollutant Diagnostics**: Benchmark concentrations against India NAAQS standards
- ⚡ **What-If Simulation**: Test policy impact (e.g. 30% PM reduction) on next-day AQI
- 🧠 **Architecture & Metrics**: Pipeline walkthrough and benchmark metrics

### 3 — Run the FastAPI Inference API

```bash
uvicorn app.predictor:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive Swagger UI.

### 4 — Run tests

```bash
pytest tests/ -v
```

---

## Docker

### Build & run locally

```bash
docker compose up --build
```

The API will be available at <http://localhost:8000>.

### Pull the published image

```bash
docker pull ghcr.io/<your-github-username>/aqi_predictor:latest
docker run -p 8000:8000 ghcr.io/<your-github-username>/aqi_predictor:latest
```

---

## CI/CD — GitHub Actions

The workflow in [`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml)
runs automatically on every push to `main`/`master`.

```
push to main
     │
     ▼
┌─────────────┐     ┌──────────────────────────┐
│  Lint & Test │────▶│  Build & Push to GHCR    │
│  (ruff, pytest)│   │  (linux/amd64 + arm64)   │
└─────────────┘     └──────────────────────────┘
                              │
                              ▼
                    ghcr.io/<owner>/aqi_predictor
                       :latest  :sha-<short>
```

**No secrets needed** — the workflow uses the built-in `GITHUB_TOKEN`.

> Make sure your repository has **"Read and write permissions"** enabled for
> `GITHUB_TOKEN`:  
> _Settings → Actions → General → Workflow permissions → Read and write_

---

## API reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

**Request body** (all pollutant fields are optional — the pipeline imputes them):

```json
{
  "City": "Delhi",
  "PM2.5": 120.0,
  "PM10": 210.0,
  "AQI": 280.0,
  "aqi_lag1": 270.0,
  "aqi_lag2": 265.0,
  "aqi_lag3": 260.0,
  "aqi_delta1": 5.0,
  "aqi_roll7_mean": 271.4,
  "aqi_roll14_mean": 268.0,
  "aqi_roll30_mean": 260.0,
  "aqi_roll7_std": 8.3,
  "month_sin": -0.5,
  "month_cos": 0.866,
  "dayofweek": 2,
  "is_weekend": 0,
  "quarter": 4
}
```

**Response:**

```json
{
  "city": "Delhi",
  "predicted_next_day_aqi": 273.45,
  "aqi_category": "Poor"
}
```

| AQI Range | Category |
|---|---|
| 0 – 50 | Good |
| 51 – 100 | Satisfactory |
| 101 – 200 | Moderate |
| 201 – 300 | Poor |
| 301 – 400 | Very Poor |
| 401 – 500 | Severe |
