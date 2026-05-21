from typing import Generator
from openai import OpenAI
from config import Config


class Provider:
    """OpenAI"""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model

    def stream_response(self, messages: list) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, temperature=0.7
        )
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is not None:
                yield token
