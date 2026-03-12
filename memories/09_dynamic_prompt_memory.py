# TYPE: Dynamic System Prompt Using Memory
# DESCRIPTION: Use @dynamic_prompt middleware to build the system prompt
# from runtime context or state — personalizing AI behaviour per user.
# The prompt is re-generated before EVERY model call using current data.
# Great for addressing users by name or adjusting tone based on their profile.

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

# --- Define what context looks like ---
class CustomContext(TypedDict):
    user_name: str

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"

# --- Build prompt dynamically using context ---
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    return f"You are a helpful assistant. Always address the user as {user_name}."

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    middleware=[personalized_prompt],   # ← runs before every model call
    context_schema=CustomContext,
)

# John gets a personalised reply
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]},
    context=CustomContext(user_name="John Smith"),
)
print(result["messages"][-1].content)
# → "Hi John Smith! The weather in Paris is always sunny!"

# Alice gets a personalised reply — same agent, different context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]},
    context=CustomContext(user_name="Alice"),
)
print(result["messages"][-1].content)
# → "Hi Alice! The weather in Tokyo is always sunny!"

# WHY USEFUL:
# Without this: same cold generic reply for every user
# With this: agent addresses each user by name automatically