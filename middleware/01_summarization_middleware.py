"""
TOPIC: SummarizationMiddleware

WHAT IT DOES:
    Automatically compresses old conversation history into a summary when
    the context window approaches its token limit. Recent messages are kept
    intact; older ones are replaced by a single summary message.

WHY THIS MATTERS:
    LLMs have a fixed context window. Without summarization, long conversations
    eventually exceed the limit and the oldest messages are silently dropped
    (or the call fails). Summarization keeps the most relevant recent context
    while preserving the gist of older history.

HOW IT WORKS:
    1. After each model turn, SummarizationMiddleware counts tokens in the
       conversation history.
    2. When total tokens exceed `trigger`, it calls a smaller/cheaper model
       to summarize everything EXCEPT the last `keep` messages.
    3. The summary replaces the old messages. The conversation continues.

CONFIGURATION:
    model   → which model writes the summary (use a fast/cheap one)
    trigger → ("tokens", N)    — summarize when N tokens are in context
              ("messages", N)  — summarize when N messages are in context
    keep    → ("messages", N)  — always keep the last N messages verbatim
              ("tokens", N)    — keep messages that fit within N tokens

WHEN TO USE:
    Long customer service chats
    Multi-turn research sessions
    Any agent loop that runs for many steps
"""

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware


# ---------------------------------------------------------------------------
# 1. Define some example tools
# ---------------------------------------------------------------------------
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [result 1, result 2, result 3]"


def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 2. Create agent with SummarizationMiddleware
# ---------------------------------------------------------------------------
agent = create_agent(
    model="gpt-4.1",
    tools=[search_web, calculate],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",          # cheaper model writes summaries
            trigger=("tokens", 4000),       # start summarizing at 4000 tokens
            keep=("messages", 20),          # always keep last 20 messages intact
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3. Run a multi-turn conversation
#    (In a real app this would be many turns — here we simulate a few)
# ---------------------------------------------------------------------------
print("=" * 60)
print("SummarizationMiddleware — long conversation demo")
print("=" * 60)

messages = []

turns = [
    "What is the capital of France?",
    "And what is 1234 * 5678?",
    "Search the web for latest AI news.",
    "What did we discuss earlier about France?",
]

for user_input in turns:
    messages.append({"role": "user", "content": user_input})

    result = agent.invoke({"messages": messages})

    # Get the last assistant message
    last_msg = result["messages"][-1]
    print(f"\nUser   : {user_input}")
    print(f"Agent  : {last_msg.content[:200]}")

    # Update our local messages list with the full updated history
    messages = result["messages"]

print(f"\nTotal messages in history: {len(messages)}")
print("(Older messages have been summarized if token limit was reached)")

# ---------------------------------------------------------------------------
# EXPECTED BEHAVIOUR:
#
#   First few turns: normal conversation
#   Once token count crosses 4000: a SummaryMessage replaces old messages
#   The agent can still answer "what did we discuss earlier" because
#   the summary preserved that context.
# ---------------------------------------------------------------------------