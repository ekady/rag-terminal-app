import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.graph.message import add_messages
from config import Config


@tool
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information. Use this tool for:
    - Pricing, costs, rates
    - Current news or events
    - Technical specifications
    - Any factual data that changes frequently
    """
    mock_results = {
        "aws ec2 pricing": "AWS EC2 t3.medium: $0.0416/hour (us-east-1, on-demand, as of latest pricing page).",
        "ec2": "AWS EC2 pricing varies by instance type. t3.medium: $0.0416/hour, t3.large: $0.0832/hour.",
        "pricing": "AWS EC2 t3.medium: $0.0416/hour in us-east-1 region.",
    }
    query_lower = query.lower()
    for k, v in mock_results.items():
        if k in query_lower:
            return v
    return f"Web search result for: {query}"


@tool
def document_retriever(query: str) -> str:
    """
    Search ONLY internal company documents, HR policies, contracts,
    or private knowledge base. Use this for internal/proprietary info.
    Do NOT use for public information like pricing, news, or specs.
    """
    mock_score = 0.41
    mock_content = "No matching internal document found."
    return f"[score: {mock_score}] {mock_content}"


tools = [web_search, document_retriever]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
llm_with_tools = llm.bind_tools(tools)

# Graph State Agent


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    """The 'reason' step — LLM decides whether to call a tool or answer."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Execute whichever tools was chosen
tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

# Run as main
if __name__ == "__main__":
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the latest AWS pricing for EC2 t3.medium instances?"
                )
            ]
        }
    )
    print(result["messages"][-1].content)
