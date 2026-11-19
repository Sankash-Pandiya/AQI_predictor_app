# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Security: run as non-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Install OS-level dependencies (required by scikit-learn / numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Final image ─────────────────────────────────────────────────────────────
FROM deps AS final

# Copy app source and model
COPY app/            ./app/
COPY aqi_pipeline.pkl .

# Ownership to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health-check: hit /health every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
CMD ["uvicorn", "app.predictor:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
