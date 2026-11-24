from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from basket.service import BasketService

WEB = Path(__file__).resolve().parents[1] / "web"
app = FastAPI(title="Basket", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")

_service: BasketService | None = None


def get_service() -> BasketService:
    global _service
    if _service is None:
        _service = BasketService()
    return _service


class CartRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    service = get_service()
    return {
        "ok": True,
        "products": len(service.catalog.products),
        "recipes": len(service.catalog.recipes),
        "backend": service.backend,
    }


@app.post("/api/cart")
def cart(body: CartRequest):
    try:
        return get_service().build(body.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "basket.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
