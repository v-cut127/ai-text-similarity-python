from fastapi import FastAPI
from pydantic import BaseModel

from models.embedding_model import model

from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class CompareRequest(BaseModel):
    text1: str
    text2: str


@app.post("/compute-embeddings")
def compute_embeddings(data: CompareRequest):

    embedding1 = model.encode([data.text1])

    embedding2 = model.encode([data.text2])

    similarity = cosine_similarity(
        embedding1,
        embedding2
    )[0][0]

    return {
        "similarity": float(similarity)
    }