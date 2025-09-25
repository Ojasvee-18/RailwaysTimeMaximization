"""
FastAPI app exposing train data and optimizer endpoints.
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
from .api import trains, optimizer, health


def create_app() -> FastAPI:
    app = FastAPI(title="Rail Traffic AI Dashboard")
    
    # Include API routers
    app.include_router(trains.router, prefix="/api/trains", tags=["trains"])
    app.include_router(optimizer.router, prefix="/api/optimizer", tags=["optimizer"])
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    
    # Serve static files (CSS, JS, images)
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    # Serve the main dashboard
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        frontend_file = frontend_dir / "index.html"
        if frontend_file.exists():
            with open(frontend_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Dashboard not found</h1>")
    
    return app


app = create_app()


