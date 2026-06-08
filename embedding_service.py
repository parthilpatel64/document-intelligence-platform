from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

model = SentenceTransformer(
    EMBEDDING_MODEL
)

def generate_embeddings(texts):

    return model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()


def generate_query_embedding(question):

    return model.encode(
        question,
        normalize_embeddings=True
    ).tolist()