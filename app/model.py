from sentence_transformers import SentenceTransformer

# Load once at startup (VERY IMPORTANT for speed)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text)