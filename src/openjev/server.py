from fastapi import FastAPI

from .model import OpenJev
from .schema import Request


def create_app(checkpoint, device="auto"):
    model = OpenJev(checkpoint, device)
    app = FastAPI(title="OpenJev", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model.config.get("release_name", "OpenJev-v0.1")}

    @app.post("/v1/decide")
    def decide(request: Request):
        return model.predict(request.state, request.questions)

    return app
