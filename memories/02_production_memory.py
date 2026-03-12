# TYPE: Production Memory (PostgreSQL Checkpointer)
# DESCRIPTION: InMemorySaver loses all data when the program stops.
# For real apps, use PostgresSaver — saves memory to a database permanently.
# Memory survives server restarts, crashes, and deployments.
# Setup: pip install langgraph-checkpoint-postgres

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # creates the required tables in PostgreSQL (run once)

    agent = create_agent(
        "openai:gpt-4.1",
        tools=[],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "user-alice-session-1"}}

    # Day 1: Alice tells the agent her name
    agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice."}]},
        config,
    )

# --- Program stops, server restarts ---
# --- Later: resume the SAME conversation ---

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    agent = create_agent(
        "openai:gpt-4.1",
        tools=[],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "user-alice-session-1"}}  # same thread_id!

    # Day 2: agent still remembers Alice
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name?"}]},
        config,
    )
    print(result["messages"][-1].content)
    # → "Your name is Alice!"  ← even after restart

# COMPARISON:
# InMemorySaver  → fast, easy, data lost on restart    → use for dev/testing
# PostgresSaver  → persistent, survives restarts       → use for production