import tiktoken

from config import (
    MAX_CHUNK_TOKENS,
    OVERLAP_TOKENS
)

encoding = tiktoken.encoding_for_model(
    "text-embedding-3-small"
)

def token_count(text):

    return len(
        encoding.encode(text)
    )

def create_chunks(
        text,
        page,
        document_name
):
    
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        candidate = (current_chunk + "\n\n" + paragraph)

        if (token_count(candidate) <= MAX_CHUNK_TOKENS):
            current_chunk = candidate
        else:
            chunks.append({
                "text": current_chunk,
                "page": page,
                "document": document_name
            })

            tokens = encoding.encode(
                current_chunk
            )

            overlap_text = encoding.decode(
                tokens[-OVERLAP_TOKENS:]
            )

            current_chunk = (
                overlap_text +
                "\n\n" +
                paragraph
            )

    if current_chunk:

        chunks.append({
            "text": current_chunk,
            "page": page,
            "document": document_name
        })

    return chunks