# TYPE: Streaming
# Watch the agent think step by step in real time,
# instead of waiting for the full answer at the end.

from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Search results for: {query}"

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search],
)

print("--- Streaming agent steps ---\n")

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Search for AI news and give me a summary"}]},
    stream_mode="values",
):
    latest = chunk["messages"][-1]

    if isinstance(latest, HumanMessage):
        print(f"User:  {latest.content}")

    elif isinstance(latest, AIMessage):
        if latest.content:
            print(f"Agent: {latest.content}")
        elif latest.tool_calls:
            names = [tc["name"] for tc in latest.tool_calls]
            print(f"Calling tools: {names}")

    elif isinstance(latest, ToolMessage):
        print(f"Tool result: {latest.content[:80]}...")

# Example output:
# User:  Search for AI news and give me a summary
# Calling tools: ['search']
# Tool result: Search results for: AI news...
# Agent: Here's a summary of the latest AI news...