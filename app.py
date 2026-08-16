"""Hugging Face Spaces default entry file.

Gradio Spaces look for ``app.py`` by default. This repo also has an ``app/``
*package* (FastAPI), so ``import app`` resolves to that package — not this file.
HF runs this file by path; we load ``gradio_app.py`` explicitly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "gradio_app.py"
_spec = importlib.util.spec_from_file_location("arxiv_rag_gradio", _path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load Gradio app from {_path}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
demo = _mod.demo

if __name__ == "__main__":
    demo.launch()
