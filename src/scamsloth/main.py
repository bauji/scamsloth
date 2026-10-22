from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def root():
    return {"status": "ScamSloth will save you"}