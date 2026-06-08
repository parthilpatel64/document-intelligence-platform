import uuid

from qdrant_client import (
    QdrantClient
)

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    VECTOR_DIMENSION
)

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    VECTOR_DIMENSION
)

client = QdrantClient(
    url=QDRANT_URL
)

def create_collection():

    collections = client.get_collections()

    existing = [
        c.name
        for c in collections.collections
    ]

    if COLLECTION_NAME not in existing:

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=
            VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE
            )
        )

def store_chunks(
        chunks,
        embeddings
):
    
    points = []

    for chunk, vector in zip(
            chunks,
            embeddings):
        
         points.append(

            PointStruct(

                id=str(uuid.uuid4()),

                vector=vector,

                payload={
                     "text":
                    chunk["text"],

                    "page":
                    chunk["page"],

                    "document":
                    chunk["document"]
                }
            )
        )
         
    client.upsert(
        collection_name=
        COLLECTION_NAME,

        points=points
    )

def search(
        query_vectors,
        top_k=5
):
    
    response = client.query_points(

        collection_name=
        COLLECTION_NAME,

        query=query_vectors,

        limit=top_k
    )

    results = response.points

    return [point for point in results if point.score > 0.70]
