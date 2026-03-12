# TYPE: Basic Tool Definition
# DESCRIPTION: The simplest way to make a tool — just add @tool above a function.
# The docstring becomes the tool's description (tells the AI what the tool does).
# Type hints are REQUIRED — they tell the AI what inputs the tool expects.
# The AI reads the name + docstring to decide when and how to call the tool.

from langchain.tools import tool
from langchain.agents import create_agent

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    # your real database logic would go here
    return f"Found {limit} results for '{query}'"

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22°C in {city}."

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Use for any calculations."""
    return str(eval(expression))

# --- Use tools in an agent ---
agent = create_agent("openai:gpt-4.1", tools=[search_database, get_weather, calculate])

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather in Paris and what is 15 * 8?"}]
})
print(result["messages"][-1].content)
# Agent calls get_weather("Paris") and calculate("15*8") automatically

# KEY RULES:
# 1. Always add type hints (query: str, limit: int)
# 2. Always write a clear docstring — the AI reads this to decide when to use the tool
# 3. Use snake_case names (search_database not "Search Database")