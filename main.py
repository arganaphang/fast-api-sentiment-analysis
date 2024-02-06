from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.get("/")
def index():
    return {"message": "OK"}


@app.get("/say")
def say(q: str | None = None):
    if q is None:
        return {"message": "Please give me some word(s)"}
    result = classifier(q)
    return {"message": result}
