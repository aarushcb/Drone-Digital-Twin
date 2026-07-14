from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Drone Digital Twin API")

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Drone Digital Twin Backend Running"
    }