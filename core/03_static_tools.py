# TYPE: Static Tools
# Fixed set of tools the agent can always use.
# Agent decides when and how to call them.

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    return str(eval(expression))

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search, get_weather, calculate],
)

result = agent.invoke({
    "messages": [
        SystemMessage(content="You are a concise, helpful assistant."),
        HumanMessage(content="What's the capital of Japan?"),
        AIMessage(content="The capital of Japan is Tokyo."),
        HumanMessage(content="Weather in Paris and what is 15 * 8?"),
    ]
})

print("━" * 54)
print("  AGENT RESULT")
print("━" * 54)

print(f"output : {result['output']}")

print("─" * 54)
print("messages:")
for msg in result["messages"]:
    print(f"  [{msg.type:<9}] {msg.content or '<tool call>'}")

print("─" * 54)
print("tool steps:")
for action, obs in result["intermediate_steps"]:
    print(f"tool : {action.tool}")
    print(f"input : {action.tool_input}")
    print(f"result : {obs}")
    print()
    
print("━" * 54)

# Agent calls get_weather("Paris") AND calculate("15*8") on its own
