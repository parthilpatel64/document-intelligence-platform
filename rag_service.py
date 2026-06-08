from groq import Groq
import os

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def build_context(results):

    context = []

    sources = []
    seen = set()

    for result in results:

        context.append(
            result.payload["text"]
        )

        source = (
            result.payload["document"],
            result.payload["page"]
        )

        if source not in seen:

            sources.append({
                "document":
                result.payload["document"],

                "page":
                result.payload["page"],

                "score":
                round(result.score, 3)
            })

            seen.add(source)


    return (
        "\n\n".join(context),
        sources
    )

def generate_answer(
        question,
        context
):

    prompt = f"""
You are a PDF research assistant.

Rules:

- Answer only using the supplied context.
- Do not use outside knowledge.
- If the answer is partially present,
  provide the partial answer.
- Never say "No content found"
  if relevant information exists.
- Merge information across chunks.
- Mention source document names.
- If the answer truly does not exist,
  say:

  "The uploaded documents do not contain
   enough information to answer this question."

Context:
{context}

Question:
{question}
"""

    response = (
        client.chat.completions.create(

            model=
            "llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )
    )

    return (
        response
        .choices[0]
        .message.content
    )