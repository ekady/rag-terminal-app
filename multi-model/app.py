from openai import OpenAI
from anthropic import Anthropic
from config import Config
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "user": "bold magenta",
        "assistant": "bold blue",
    }
)

console = Console(theme=custom_theme)

openai = OpenAI(api_key=Config.OPENAI_API_KEY)
anthropic = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

GPT_SYSTEM_PROMPT = """You are an ambitious, optimistic futurist who argues with \
enthusiasm and data. You believe technology and human ingenuity can solve almost \
any problem. In debates, you make bold claims, cite hypothetical statistics, and \
challenge your opponent to think bigger. Keep each response to 3-4 sentences — \
punchy and direct."""

CLAUDE_SYSTEM_PROMPT = """You are a careful, nuanced philosopher who values \
wisdom over hype. You push back on overconfidence, highlight second-order \
consequences, and remind your opponent what history teaches us. You're not a \
pessimist — you simply believe good ideas survive scrutiny. Keep each response \
to 3-4 sentences — sharp and thoughtful."""


def flip_roles(messages: list[dict]) -> list[dict]:
    new_messages = []
    for msg in messages:
        if msg["role"] == "system":
            new_messages.append(msg)
        elif msg["role"] == "user":
            new_messages.append({**msg, "role": "assistant"})
        elif msg["role"] == "assistant":
            new_messages.append({**msg, "role": "user"})
    return new_messages


def call_openai(history):
    messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        *history,
    ]
    response = openai.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=messages,
        temperature=0.9,
    )
    return response.choices[0].message.content


def call_claude(history):
    claude_perspective = flip_roles(history)

    response = anthropic.messages.create(
        model=Config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=CLAUDE_SYSTEM_PROMPT,
        messages=claude_perspective,
        temperature=0.9,
    )
    return response.content[0].text


def llm_conversation(topic: str, rounds: int = 10):
    history = [
        {
            "role": "user",
            "content": f"Let's have a debate on the topic: {topic}. You go first.",
        },
    ]

    console.print(f"Starting LLM Conversation on topic: {topic}...", style="info")
    console.print(f"Debate Topic: {topic}", style="info")
    console.print("\n\n" + "=" * 80)

    for i in range(rounds):
        openai_resp = call_openai(history)
        claude_resp = call_claude(history)
        history.append({"role": "assistant", "content": openai_resp})
        history.append({"role": "assistant", "content": claude_resp})
        console.print(f"\n\nRound {i + 1}", style="info")
        console.print("=" * 80, style="info")
        console.print(f"OpenAI: {openai_resp}", style="user")
        console.print(f"Claude: {claude_resp}", style="assistant")
        console.print("=" * 80, style="info")


if __name__ == "__main__":
    llm_conversation("Will AI lead to human extinction?", 2)
