"""
TOPIC: ModelCallLimitMiddleware

WHAT IT DOES:
    Caps the total number of model (LLM) API calls an agent can make,
    either per conversation thread or per single run. When the limit is
    reached, the agent stops gracefully (or raises, depending on config).

WHY THIS MATTERS:
    Without limits, a looping or confused agent can rack up thousands of
    API calls, costing real money and potentially running forever.

HOW IT WORKS:
    - thread_limit: max calls ever on a given thread_id (across all runs)
    - run_limit:    max calls in a single agent.invoke() / agent.stream()
    - exit_behavior:
        "end"   → silently stop, return whatever state we have
        "error" → raise an exception

REQUIRES:
    A checkpointer when using thread_limit (to track call counts across runs).

CONFIGURATION:
    ModelCallLimitMiddleware(
        thread_limit=10,    # max 10 model calls total for this thread
        run_limit=5,        # max 5 per single .invoke()
        exit_behavior="end" # or "error"
    )

WHEN TO USE:
    Production cost controls
    Preventing infinite reasoning loops
    Enforcing SLAs on agent response time
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------------------------
# 1. Tool
# ---------------------------------------------------------------------------
def think_step(step: int) -> str:
    """Simulate a multi-step reasoning tool."""
    return f"Step {step} complete. Still working..."


# ---------------------------------------------------------------------------
# 2. Agent with strict model call limits
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()  # required for thread_limit tracking

agent = create_agent(
    model="gpt-4.1",
    tools=[think_step],
    checkpointer=checkpointer,
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,        # this thread can call the model at most 10 times total
            run_limit=3,            # each single .invoke() gets at most 3 model calls
            exit_behavior="end",    # stop gracefully when limit hit
        ),
    ],
)

config = {"configurable": {"thread_id": "limited_thread_01"}}


# ---------------------------------------------------------------------------
# 3. Run — limit will be enforced
# ---------------------------------------------------------------------------
print("=" * 60)
print("ModelCallLimitMiddleware — run_limit=3 demo")
print("=" * 60)

result = agent.invoke(
    {"messages": [{"role": "user", "content":
        "Please run think_step 10 times and summarize the results."}]},
    config=config,
    version="v2",
)

# Count how many AI messages were generated
ai_messages = [m for m in result.value["messages"]
               if type(m).__name__ == "AIMessage"]
print(f"Model calls made   : {len(ai_messages)}")
print(f"(Limited to run_limit=3 per run)")
print(f"\nFinal agent message: {result.value['messages'][-1].content[:300]}")


# ---------------------------------------------------------------------------
# 4. Show thread-level tracking: second run on the same thread
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Second .invoke() on same thread (thread_limit accumulates)")
print("=" * 60)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Do one more step."}]},
    config=config,
    version="v2",
)
print(f"Second run completed. Thread total calls: {len(ai_messages) + 1}+")
print(f"Response: {result2.value['messages'][-1].content[:200]}")

# ---------------------------------------------------------------------------
# EXPECTED BEHAVIOUR:
#
#   Run 1: Agent tries to call think_step 10 times but hits run_limit=3
#          → agent stops after 3 model calls, returns partial result
#   Run 2: Thread counter continues from where it left off
#          → once thread_limit=10 is hit, no more calls on this thread
# ---------------------------------------------------------------------------