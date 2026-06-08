from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_results(
        question,
        search_results,
        top_n=5
):

    pairs = []

    for result in search_results:

        pairs.append(
            (
                question,
                result.payload["text"]
            )
        )

    scores = reranker.predict(pairs)

    ranked = list(
        zip(search_results, scores)
    )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item[0]
        for item in ranked[:top_n]
    ]