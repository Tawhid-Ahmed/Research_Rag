# Hugging Face Spaces (Docker Space) image for the free live demo.
# Runs the FastAPI backend and Streamlit UI in one container on the free CPU
# tier using open models + Chroma (no secrets required).
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Build context is the repo root `arxiv-rag/` (see huggingface/README.md).
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY eval ./eval
COPY huggingface/start.sh ./start.sh
RUN chmod +x ./start.sh

# Spaces routes external traffic to port 7860.
EXPOSE 7860
ENV API_BASE_URL=http://localhost:8000

CMD ["./start.sh"]
