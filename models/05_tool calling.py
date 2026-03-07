# TYPE: Tool Calling
# DESCRIPTION: Tools let the model call external functions (like an API or database).
# The model decides WHEN and HOW to call the tool based on the question.
# You bind tools to the model, then the model requests tool calls in its response.

from langchain.chat_models import init_chat_model
from langchain.tools import tool

model = init_chat_model("gpt-4.1", model_provider="openai")

@tool
def get_weather(location: str) -> str:
    """Get the current weather at a location."""
    return f"It's sunny and 22°C in {location}."

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"{city} has a population of 2.1 million."


# --- Bind tools to model ---
model_with_tools = model.bind_tools([get_weather, get_population])

# --- Model decides which tool to call ---
response = model_with_tools.invoke("What's the weather like in Boston?")

# Model doesn't answer yet — it REQUESTS a tool call first
for tool_call in response.tool_calls:
    print(f"Tool requested : {tool_call['name']}")   # → get_weather
    print(f"With arguments : {tool_call['args']}")   # → {"location": "Boston"}

# NOTE: When using create_agent(), the agent loop runs tools automatically.
# When using model standalone (like here), YOU must run the tool
# and pass the result back manually.
# For most use cases, use create_agent() — it handles all of this for you.