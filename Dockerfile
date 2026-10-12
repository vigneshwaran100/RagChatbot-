# ─────────────────────────────────────────────────────────────────────────────
# RAG Knowledge Assistant — App Dockerfile
# Connects to an EXISTING Qdrant instance (no Qdrant service here).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for docling (PDF processing via pdfminer, PIL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY .env.example .env.example

# Create upload temp directory
RUN mkdir -p /app/uploads

# Non-root user for security
RUN adduser --disabled-password --gecos "" raguser && chown -R raguser /app
USER raguser

# Pre-download the MPNet model at build time to avoid cold-start delay
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
