---
title: arXiv RAG Assistant
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: RAG over arXiv AI/ML papers with citations
---

# arXiv RAG Assistant — Live Demo (Hugging Face Spaces)

**Free tier:** use the **Gradio** SDK. On current HF “New Space” screens, Docker is
marked **Paid** and Streamlit may not appear — choose **Gradio → Blank**.

`app.py` starts the FastAPI backend in the background, then serves Chat + Admin.

## Create the Space

1. [New Space](https://huggingface.co/new-space) → **Manual setup**
2. Name e.g. `Arxiv-Research-RAG`
3. **SDK: Gradio** (not Docker)
4. Template: **Blank**
5. Hardware: **Free tier / CPU basic**
6. Create, then **Settings → Connect GitHub** to this repo  
   (or upload/push so `gradio_app.py` + `requirements.txt` are at the Space root).  
   Set the Gradio **app file** to `gradio_app.py` (not `app.py` — that name
   conflicts with our `app/` Python package).
7. Secrets (Settings → Variables and secrets):

| Name | Value |
|------|--------|
| `HUGGINGFACE_API_KEY` | free HF token ([create token](https://huggingface.co/settings/tokens)) |
| `LLM_PROVIDER` | `huggingface` |
| `LLM_MODEL` | e.g. `HuggingFaceH4/zephyr-7b-beta` |

Optional: `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.

## First-run corpus

Vector store starts empty. **Admin** tab → ingest `1706.03762`, then **Chat**:  
`What is multi-head attention?`

## Local Streamlit

`streamlit run ui/app.py` (or `space_app.py`) still works locally with a separate
`uvicorn` process. The public free Space path is Gradio.

See the main [`README.md`](../README.md).
