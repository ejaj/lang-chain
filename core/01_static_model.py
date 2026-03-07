# TYPE: Static Model
# Use one fixed model for the whole agent. Best for simple, predictable projects.

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# --- Option A: shortcut string ---
# agent = create_agent("openai:gpt-4.1", tools=[])

# --- Option B: full control ---
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.2,  # 0 = focused, 1 = creative
    max_tokens=1000,
    timeout=30,
)
agent = create_agent(model, tools=[])

# --- Run ---
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is 42 * 17?"}]
})
print(result["messages"][-1].content)
# → "42 × 17 = 714"