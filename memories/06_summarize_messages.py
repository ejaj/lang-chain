# TYPE: Summarize Messages (SummarizationMiddleware)
# DESCRIPTION: Instead of deleting old messages (losing info),
# summarize them into a short paragraph and keep that instead.
# Best of both worlds: saves space AND keeps important facts.
# Uses a cheap/fast model to summarize, keeping costs low.
# trigger = when to summarize (token count)
# keep    = how many recent messages to keep as-is after summarizing

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

checkpointer = InMemorySaver()

agent = create_agent(
    model="openai:gpt-4.1",           # main smart model for answering
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",  # cheap model just for summarizing
            trigger=("tokens", 4000),      # summarize when history hits 4000 tokens
            keep=("messages", 20),         # keep last 20 messages as-is after summary
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "Hi, my name is Bob."}, config)
agent.invoke({"messages": "Write a short poem about cats."}, config)
agent.invoke({"messages": "Now write one about dogs."}, config)

result = agent.invoke({"messages": "What is my name?"}, config)
result["messages"][-1].pretty_print()
# → "Your name is Bob!"
# ← even after old messages were summarized, the key fact "Bob" was kept

# COMPARISON of 3 approaches:
#
# Trim    → fast, cheap, loses info  → use when old messages don't matter
# Delete  → pick exactly what to remove → use for specific cleanup rules
# Summarize → keeps facts, costs a bit more → USE THIS for most production apps