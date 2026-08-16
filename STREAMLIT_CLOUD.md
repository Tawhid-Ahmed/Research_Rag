# Streamlit Community Cloud

## Deploy settings

| Field | Value |
|--------|--------|
| Repository | `Tawhid-Ahmed/Research_Rag` |
| Branch | `deploy/streamlit-cloud` |
| Main file path | `space_app.py` |
| **Python version** | **3.12** (Advanced settings) |

Python **must** be 3.12. Community Cloud currently defaults to 3.14, which cannot install this app’s pinned `tiktoken` / `pydantic-core` / Chroma `tokenizers` wheels.

After changing Python version: **Save** → **Reboot app**. If logs still say `Using Python 3.14`, delete the app and redeploy, selecting 3.12 in Advanced settings before Deploy.

## Secrets

Copy needed keys from `.env` into **Advanced settings → Secrets** (TOML), for example:

```toml
LLM_PROVIDER = "huggingface"
HF_TOKEN = "hf_..."
HF_MODEL = "..."
VECTOR_STORE = "chroma"
ENVIRONMENT = "production"
```
