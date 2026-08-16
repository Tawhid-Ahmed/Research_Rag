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

1. Create a fine-grained token at https://huggingface.co/settings/tokens with
   **Make calls to Inference Providers**.
2. App settings → **Advanced settings** → **Secrets** (TOML). Root-level keys
   become env vars for the API process:

```toml
LLM_PROVIDER = "huggingface"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = "hf_your_token_here"
VECTOR_STORE = "chroma"
ENVIRONMENT = "production"
```

3. **Save**, then **Reboot** the app (needed so uvicorn restarts with the new env).
4. Sidebar should show `HF token: set`. If it still says `missing`, reboot again
   after saving secrets.
