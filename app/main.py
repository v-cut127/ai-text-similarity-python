from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.schemas import CompareRequest
from app.model import get_embedding

app = FastAPI()

@app.post("/compute-embeddings")
def compute_embeddings(data: CompareRequest):

    if not data.text1.strip() or not data.text2.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        emb1 = get_embedding(data.text1)
        emb2 = get_embedding(data.text2)

        # reshape for sklearn
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)

        score = cosine_similarity(emb1, emb2)[0][0]

        return {
            "similarity": float(score)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))