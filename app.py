from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf_processor import extract_pages
from chunking_service import create_chunks
from embedding_service import generate_embeddings,generate_query_embedding
from qdrant_service import store_chunks
from models import QueryRequest,QueryResponse
import os
from qdrant_service import (
    create_collection,
    store_chunks,
    search
)
from rag_service import build_context, generate_answer
from reranker_service import (
    rerank_results
)


app = FastAPI(
    title="Multi PDF Research Assistant"
)

# Use absolute path to avoid issues with relative paths when running with uvicorn
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "message": "Multi PDF Research Assistant Running"
    }

@app.on_event("startup")
def startup():

    create_collection()

    print(
        "Qdrant collection ready"
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:
        file_content = await file.read()
        print(f"DEBUG: File size read: {len(file_content)} bytes")
        print(f"DEBUG: Writing to: {file_path}")
        
        with open(file_path, "wb") as buffer:
            bytes_written = buffer.write(file_content)
            print(f"DEBUG: Bytes written: {bytes_written}")
            buffer.flush()
            os.fsync(buffer.fileno())
        
        # Verify file was written
        if not os.path.exists(file_path):
            raise Exception(f"File was not saved to disk at {file_path}")
        
        file_size = os.path.getsize(file_path)
        print(f"DEBUG: Verified file exists with size: {file_size} bytes")

        pages = extract_pages(file_path)

        all_chunks =[]

        for page in pages:

            page_chunks = create_chunks(
                text=page["text"],
                page=page["page"],
                document_name=file.filename
            )

            all_chunks.extend(
                page_chunks
            )

        if len(all_chunks) == 0:

                raise HTTPException(
                status_code=400,
                detail=
                "No text found in PDF"
                )
    
        chunk_texts = [

            chunk["text"]

            for chunk in all_chunks
        ]
    
        embeddings = generate_embeddings(
            chunk_texts
        )
    
        store_chunks(
            all_chunks,
            embeddings
        )
    
        return {

            "status": "success",

            "file":
            file.filename,

            "pages":
            len(pages),

            "chunks":
            len(all_chunks)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/query")
def query_documents(
        request: QueryRequest
):

    try:

        question = request.question

        query_vectors = (
            generate_query_embedding(
                question
            )
        )

        results = search(
            query_vectors=query_vectors,
            top_k=20
        )

        if len(results) == 0:

            return {

                "answer":
                "No relevant content found.",

                "sources": []
            }

        results = rerank_results(
            question=question,
            search_results=results,
            top_n=5
        )

        context, sources = (
            build_context(results)
        )

        answer = generate_answer(
            question=question,
            context=context
        )

        return {

            "question":
            question,

            "answer":
            answer,

            "sources":
            sources
        }
    
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )