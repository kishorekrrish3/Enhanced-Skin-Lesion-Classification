from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from .api.router import router

app = FastAPI(title="Skin Lesion API")

# Setup CORS for the Next.js frontend (default port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "results")
if os.path.exists(RESULTS_DIR):
    app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
