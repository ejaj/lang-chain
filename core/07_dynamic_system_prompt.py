# TYPE: Dynamic System Prompt
# Change the agent's instructions at runtime based on who is asking.
# Experts get technical answers. Beginners get simple explanations.

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict

class Context(TypedDict):
    user_role: str  # "expert" | "beginner" | "user"

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.get("user_role", "user")

    base = "You are a helpful assistant."

    if role == "expert":
        return f"{base} Use technical terms. Be precise and detailed."
    elif role == "beginner":
        return f"{base} Use simple words. Avoid jargon. Give examples."
    else:
        return base

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[],
    middleware=[role_based_prompt],
    context_schema=Context,
)

# Expert gets a deep technical answer
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain transformers"}]},
    context={"user_role": "expert"},
)

# Beginner gets a simple explanation — same question, different answer
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain transformers"}]},
    context={"user_role": "beginner"},
)