from gradio import ChatInterface

from db import init_db
from rag import ask_with_memory_and_streaming

init_db()


def chat(query: str, history: list[list[str]]):
    full_answer = ""
    for chunk, src in ask_with_memory_and_streaming(query):
        full_answer += chunk
        yield full_answer


if __name__ == "__main__":
    chat_interface = ChatInterface(
        fn=chat,
        title="RAG Terminal Assistant",
        description="Ask questions about your indexed documents",
    )
    chat_interface.launch()
