from __future__ import annotations

import os
import warnings
from pathlib import Path

# Load .env from backend/ directory if it exists (for local dev)
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Suppress Pydantic v2 protected namespace warnings for fields like model_path, model_file
# in schemas and FastAPI endpoint parameters before any modules are imported.
warnings.filterwarnings("ignore", message="Field.*has conflict with protected namespace")

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from core.firebase_auth import get_current_user  # noqa: E402
from core.firestore_db import get_client  # noqa: E402
from routers import audit, bias, colab, datasets, fixes, llm_narrative, monitoring, pipeline, sandbox, project, chat, pattern_review, mitigate, user  # noqa: E402

app = FastAPI(title="Unbiased AI Decision Platform")

_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
]

# Set CORS_ALLOW_ALL=1 in the environment to open wildcard origins for local dev.
_allow_all = os.getenv("CORS_ALLOW_ALL", "0") == "1"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _CORS_ORIGINS,
    allow_credentials=not _allow_all,  # credentials + wildcard is illegal; disable when wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth dependency applied globally to all protected routers
_auth = [Depends(get_current_user)]

# Public — project and pipeline declare auth per-endpoint (some task status/result
# endpoints are intentionally token-free because task_ids are unguessable UUIDs).
app.include_router(project.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")  # sample datasets are public

# Protected routers
app.include_router(audit.router, prefix="/api", dependencies=_auth)
app.include_router(bias.router, prefix="/api", dependencies=_auth)
app.include_router(fixes.router, prefix="/api", dependencies=_auth)
app.include_router(sandbox.router, prefix="/api", dependencies=_auth)
app.include_router(monitoring.router, prefix="/api", dependencies=_auth)
app.include_router(colab.router, prefix="/api", dependencies=_auth)
app.include_router(llm_narrative.router, prefix="/api", dependencies=_auth)
app.include_router(chat.router, prefix="/api", dependencies=_auth)
app.include_router(pattern_review.router, prefix="/api", dependencies=_auth)
app.include_router(mitigate.router, prefix="/api")  # auth declared per-endpoint
app.include_router(user.router, prefix="/api")       # auth declared per-endpoint


@app.on_event("startup")
def startup_seed() -> None:
    # Warm Firestore connection on startup
    try:
        get_client()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Firestore init failed at startup: %s", exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# Serve Frontend
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"), name="assets")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # If the path looks like a file (has an extension), don't serve index.html
    if "." in full_path.split("/")[-1]:
        file_path = os.path.join("../frontend/dist", full_path)
        if os.path.exists(file_path):
            return FileResponse(file_path)

    # Otherwise, serve index.html for client-side routing
    return FileResponse("../frontend/dist/index.html")
