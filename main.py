from fastapi import FastAPI
from api.label_extraction import router as label_router
from api.expiry_extraction import router as expiry_router
# from api.freshness_prediction import router as freshness_router

app = FastAPI(title="Image Processing API")

app.include_router(label_router)
app.include_router(expiry_router)
# app.include_router(freshness_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API!"}
