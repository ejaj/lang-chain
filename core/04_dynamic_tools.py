# TYPE: Dynamic Tools
# Show only the tools the user is allowed to use.
# Guests get public tools. Logged-in users get everything.

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.agents.middleware import wrap_model_call


@tool
def public_search(query: str) -> str:
    """Public web search — available to everyone."""
    return f"Public results for: {query}"

@tool
def private_search(query: str) -> str:
    """Private database search — logged-in users only."""
    return f"Private results for: {query}"

@tool
def admin_tool(command: str) -> str:
    """Admin actions — restricted."""
    return f"Admin: {command}"

ALL_TOOLS = [public_search, private_search, admin_tool]


@wrap_model_call
def filter_by_auth(request, handler):
    authenticated = request.state.get("authenticated", False)
    if not authenticated:
        request = request.override(
            tools=[t for t in request.tools if t.name.startswith("public_")]
        )
    return handler(request)


agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=ALL_TOOLS,
    middleware=[filter_by_auth],
)
executor = AgentExecutor(agent=agent, tools=ALL_TOOLS)


def print_result(result, label: str):
    print("━" * 54)
    print(f"  {label}")
    print("━" * 54)
    print(f"output       : {result['output']}")
    print("─" * 54)
    print("messages:")
    for msg in result["messages"]:
        print(f"  [{msg.type:<9}] {msg.content or '<tool call>'}")
    print("─" * 54)
    print("tool steps:")
    for action, obs in result["intermediate_steps"]:
        print(f"  tool   : {action.tool}")
        print(f"  input  : {action.tool_input}")
        print(f"  result : {obs}")
        print()
    print("━" * 54)
    print()


# ── Guest ─────────────────────────────────────────────
result = executor.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Search for news"),
    ],
    "state": {"authenticated": False},
})
print_result(result, "GUEST (authenticated=False)")


# ── Logged in ─────────────────────────────────────────
result = executor.invoke({
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Search private records"),
    ],
    "state": {"authenticated": True},
})
print_result(result, "LOGGED IN (authenticated=True)")