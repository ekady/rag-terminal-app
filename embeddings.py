from openai import OpenAI
from config import Config
from chunking import chunk_text

client = OpenAI(api_key=Config.OPENAI_API_KEY)


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
