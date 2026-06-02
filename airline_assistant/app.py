import json
from .tools_assistant import (
    get_ticket_price,
    get_current_bookings,
    book_flight,
    cancel_booking,
    get_destinations,
)
from openai import OpenAI
from config import Config
from rich.console import Console
from rich.theme import Theme

console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "red bold",
            "success": "green",
            "user": "bold magenta",
            "assistant": "bold blue",
        }
    )
)

client = OpenAI(api_key=Config.OPENAI_API_KEY)

tools = {
    "get_destinations": get_destinations,
    "get_ticket_price": get_ticket_price,
    "get_current_bookings": get_current_bookings,
    "book_flight": book_flight,
    "cancel_booking": cancel_booking,
}

tools_openai = [
    {
        "type": "function",
        "function": {
            "name": "get_destinations",
            "description": "Get the list of all available destinations",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination of the ticket",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Get the price of a ticket to a specific destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination of the ticket",
                    },
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_bookings",
            "description": "Get the current bookings of a passenger",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    },
                },
                "required": ["passenger_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight for a passenger",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination of the flight",
                    },
                },
                "required": ["passenger_name", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": "Cancel a booking for a passenger",
            "parameters": {
                "type": "object",
                "properties": {
                    "passenger_name": {
                        "type": "string",
                        "description": "The name of the passenger",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination of the flight",
                    },
                },
                "required": ["passenger_name", "destination"],
            },
        },
    },
]


def handle_tool_calls(messages):
    message_output = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        messages=messages,
        tools=tools_openai,
        max_tokens=5000,
        temperature=0.6,
    )
    messages.append(message_output.choices[0].message)
    last_message = messages[-1]
    while last_message.tool_calls:
        tool_calls = last_message.tool_calls
        for tool_call in tool_calls:
            tool_args = json.loads(tool_call.function.arguments)
            passenger_name = (
                tool_args.get("passenger_name")
                if tool_args.get("passenger_name")
                else None
            )
            destination = (
                tool_args.get("destination") if tool_args.get("destination") else None
            )
            tool_name = tool_call.function.name
            args = (
                [passenger_name, destination]
                if passenger_name and destination
                else [passenger_name]
                if passenger_name
                else [destination]
                if destination
                else []
            )
            tool_content = tools[tool_name](*args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_content),
                }
            )
        last_message = (
            client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                tools=tools_openai,
                max_tokens=5000,
                temperature=0.6,
            )
            .choices[0]
            .message
        )
        messages.append(last_message)

    return messages


def main():
    console.print("[bold cyan]Airline Assistant[/bold cyan]")
    console.print("[bold cyan]Type 'quit' to exit[/bold cyan]")

    messages = [
        {
            "role": "system",
            "content": "You are an airline assistant. If information provided by the user is not enough to do tool calling, always ask for the missing information.",
        }
    ]

    while True:
        user_input = console.input("[bold magenta]User: [/bold magenta]")
        if user_input == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        messages = handle_tool_calls(messages)

        console.print("[bold blue]Assistant:[/bold blue]", messages[-1].content)


if __name__ == "__main__":
    main()
