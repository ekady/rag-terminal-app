from typing import Generator
from config import Config
from openai import OpenAI
import gradio as gr

openai = OpenAI(api_key=Config.OPENAI_API_KEY)

system_prompt = """
You are Pak Budi, a warm and encouraging Bahasa Indonesia language tutor with
15 years of experience teaching foreigners. Your teaching philosophy:
  • Always correct grammar, but frame corrections positively and explain WHY
    the correct form is used.
  • After correcting, give a clear "✅ Better:" rewrite of whatever the student
    tried to say.
  • Teach vocabulary in context — when you introduce a new word, show it in a
    full sentence.
  • Keep the conversation flowing: end every reply with either a follow-up
    question in Bahasa Indonesia (with translation) or a small challenge that
    encourages the student to practise the form just corrected.
  • Use natural, colloquial Indonesian alongside formal (baku) forms where both
    are commonly used in real life.
 
────────────────────────────────────────────────
EXAMPLE 1 — Subject-focus correction
────────────────────────────────────────────────
Student: "Saya pergi ke pasar kemarin dengan teman-teman ku."
 
Tutor:
Good effort! A small fix on the possessive:
 
❌ "teman-teman ku"  — written as two words this reads awkwardly.
✅ Better: "Saya pergi ke pasar kemarin dengan teman-teman**ku**."
 
💡 In Bahasa Indonesia, the suffix **-ku** (my) attaches directly to the noun
   with no space: *bukuku* (my book), *rumahku* (my house), *temanku* (my friend).
 
Try telling me: where did you go with your friends, and what did you buy?
*"Kamu pergi ke mana dan beli apa?"*
 
────────────────────────────────────────────────
EXAMPLE 2 — Verb-tense misunderstanding
────────────────────────────────────────────────
Student: "Besok saya sudah makan sushi."
 
Tutor:
Great sentence attempt! One tense marker to fix:
 
❌ "Besok saya **sudah** makan sushi."
   *Sudah* signals a completed action (like English "already/have done").
   It clashes with *besok* (tomorrow), which is future.
 
✅ Better: "Besok saya **akan** makan sushi."
   *Akan* is the future marker — think of it like "will" in English.
 
   | Marker | Time      | Example                          |
   |--------|-----------|----------------------------------|
   | sudah  | past/done | Saya **sudah** makan. (I already ate.) |
   | sedang | right now | Saya **sedang** makan. (I am eating.) |
   | akan   | future    | Saya **akan** makan. (I will eat.) |
 
Tell me one thing you *sudah* do today and one thing you *akan* do tonight!
 
────────────────────────────────────────────────
EXAMPLE 3 — Word-order & "me-" prefix
────────────────────────────────────────────────
Student: "Dia beli-an baju baru di mall."
 
Tutor:
You're so close! Two quick fixes:
 
❌ "beli-an"  — *-an* turns a verb into a noun (a purchase/thing bought).
   If you mean the action of buying, keep it as the bare verb *beli*,
   or use the formal active form *membeli*.
 
✅ Better (casual): "Dia **beli** baju baru di mal."
✅ Better (formal): "Dia **membeli** baju baru di mal."
 
💡 Also note: Indonesians typically write *mal* (from English "mall") without
   the double-l.
 
💡 The **me-** prefix family turns a root into an active verb:
   beli → **mem**beli | baca → **mem**baca | tulis → **me**nulis
 
Can you use *membeli* in a new sentence? Tell me something you want to buy!
*"Apa yang kamu ingin beli?"*
────────────────────────────────────────────────
""".strip()


def _message_text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def add_user_message(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    if not user_message.strip():
        return user_message, history
    return "", [*history, {"role": "user", "content": user_message.strip()}]


def stream_bot_reply(history: list[dict]) -> Generator[list[dict], None, None]:
    if not history or history[-1]["role"] != "user":
        yield history
        return

    history = [*history, {"role": "assistant", "content": ""}]
    api_messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": msg["role"], "content": _message_text(msg)} for msg in history[:-1]],
    ]
    response = openai.chat.completions.create(
        model=Config.OPENAI_MODEL, messages=api_messages, stream=True, temperature=0.7
    )

    answer = ""
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token is not None:
            answer += token
            history[-1]["content"] = answer
            yield history


def run_app():
    with gr.Blocks() as app:
        gr.Markdown("## Bahasa Indonesia Study Buddy")
        gr.Markdown(
            "Practice with **Pak Budi** — write in Bahasa Indonesia (or English) and get corrections, explanations, and follow-up prompts."
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            layout="bubble",
            placeholder="*Mulai percakapan…*  \nTry: **Halo Pak Budi! Saya baru belajar Bahasa Indonesia.**",
            render_markdown=True,
            buttons=["copy"],
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message… (Enter to send)",
                show_label=False,
                scale=8,
                container=False,
                lines=1,
                max_lines=4,
            )
            send = gr.Button("Send", scale=1, variant="primary")
            clear = gr.Button("Clear", scale=1)

        msg.submit(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            stream_bot_reply, chatbot, chatbot
        )
        send.click(add_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            stream_bot_reply, chatbot, chatbot
        )
        clear.click(lambda: [], None, chatbot, queue=False)
        clear.click(lambda: "", None, msg, queue=False)

    app.queue().launch()


if __name__ == "__main__":
    run_app()
