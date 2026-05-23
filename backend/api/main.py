"""FastAPI application entrypoint.

Single-port architecture:
  /api/*   → API routes (registered first)
  /*       → frontend/dist static files + SPA fallback (registered last)

Run (dev):
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
Run (prod, after `npm --prefix ../frontend run build`):
  uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import service
from .routes import router as api_router

# frontend/dist is sibling to backend/. Build it with `npm run build` from frontend/.
FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model load is sync; running inside the async lifespan is fine because
    # it executes once at startup before any request is served.
    try:
        service.load()
    except Exception as e:
        # Surface load failures cleanly — /api/health will report model_loaded=false.
        import logging
        logging.getLogger("gnn-recommender").error(f"service.load() failed: {e!r}")
    yield


# Hide docs in production (env DISABLE_DOCS=1)
_disable_docs = os.environ.get("DISABLE_DOCS") == "1"

app = FastAPI(
    title="GNN Product Recommender",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _disable_docs else "/docs",
    redoc_url=None if _disable_docs else "/redoc",
    openapi_url=None if _disable_docs else "/openapi.json",
)

# 1) API routes first so they always win over the static catch-all below.
app.include_router(api_router)


# 2) Static + SPA fallback. Mount only if a build exists; in dev the frontend
#    is served by the Vite dev server and proxies /api here.
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str, request: Request):
        # Serve real files under dist/ if they exist (favicon, robots.txt, etc.).
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        # Otherwise hand the SPA its entry point — React Router takes over.
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def dev_landing():
        return JSONResponse(
            {
                "message": (
                    "frontend/dist not found. In dev, open the Vite server "
                    "(http://127.0.0.1:5173). For prod, run "
                    "`npm --prefix frontend run build` first."
                ),
                "api_docs": "/docs" if not _disable_docs else None,
            }
        )
