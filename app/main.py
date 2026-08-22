import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.api.v1.api import api_router

app = FastAPI(
    title="Enterprise Agentic AI Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount modular router
app.include_router(api_router, prefix="/api/v1")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()