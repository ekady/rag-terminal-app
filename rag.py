import numpy as np
from openai import OpenAI

from config import Config
from db import get_connection
from embeddings import get_embedding

client = OpenAI(api_key=Config.OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the user's question. If the context doesn't contain
relevant information, say so honestly — do not make up answers.

Always cite which file(s) your answer is based on when possible."""


def search_similar(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = get_embedding(query)
    query_array = np.array(query_embedding)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT filename, chunk_index, content,
               1 - (embedding <=> %s) AS similarity
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s;
        """,
        (query_array, query_array, top_k),
    )

    results = []
    for row in cur.fetchall():
        results.append(
            {
                "filename": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "similarity": float(row[3]),
            }
        )

    cur.close()
    conn.close()

    return results


def build_context(results: list[dict]) -> str:
    if not results:
        return "No relevant documents found."

    context_parts = []
    for r in results:
        source = f"{r['filename']} (chunk {r['chunk_index']})"
        context_parts.append(f"--- Source: {source} ---\n{r['content']}")

    return "\n\n".join(context_parts)


def ask(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    sources = search_similar(query, top_k=top_k)
    context = build_context(sources)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\n---\n\nQuestion: {query}",
        },
    ]

    response = client.chat.completions.create(
        model=Config.LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    return answer, sources
