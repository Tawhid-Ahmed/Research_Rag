---
title: arXiv RAG Assistant
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# arXiv RAG Assistant — Live Demo (Hugging Face Spaces)

This Space hosts the free live demo of the arXiv RAG Assistant. It runs the
FastAPI backend and Streamlit UI in a single Docker container on the free CPU
tier using open models (`sentence-transformers` + a Hugging Face hosted LLM) and
a file-persisted Chroma vector store — no secrets required.

## Deploying

The Space build context is the `arxiv-rag/` directory. To deploy, point the
Space at this repo and set the Dockerfile path to `huggingface/Dockerfile`, or
copy `huggingface/Dockerfile`, `huggingface/start.sh`, and this README to the
Space root.

Optional secrets (set in **Space settings → Variables and secrets**) unlock the
managed-API providers:

- `HUGGINGFACE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`

See the main [`README.md`](../README.md) for full documentation.
