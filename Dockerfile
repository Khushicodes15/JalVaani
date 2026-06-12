# ── Stage 1: Build React UI ───────────────────────────────────────────────────
FROM node:20-alpine AS ui-builder
WORKDIR /app/jalvaani_ui
COPY jalvaani_ui/package*.json ./
RUN npm ci --prefer-offline
COPY jalvaani_ui/ ./
RUN npm run build          # Output → /app/jalvaani_ui/dist


# ── Stage 2: Python API ───────────────────────────────────────────────────────
FROM python:3.11-slim AS api
WORKDIR /app

# System deps (torch needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (cached layer — invalidated only when requirements.txt changes)
COPY jalvaani_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY jalvaani_api/ ./jalvaani_api/

# Copy built UI from stage 1 into the path that STATIC_DIR points to
COPY --from=ui-builder /app/jalvaani_ui/dist ./jalvaani_ui/dist

# Model artifacts are NOT baked into the image — mount them at runtime.
# docker run -v /host/path/to/models:/app/jalvaani_api/saved_models ...
# docker run -v /host/path/to/data:/app/jalvaani_api/data ...

ENV STATIC_DIR=/app/jalvaani_ui/dist \
    CORS_ORIGINS=http://localhost:8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

WORKDIR /app/jalvaani_api
CMD ["gunicorn", "-c", "../gunicorn.conf.py", "main:app"]
