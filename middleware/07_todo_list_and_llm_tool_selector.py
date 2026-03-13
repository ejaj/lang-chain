"""
TOPIC: TodoListMiddleware + LLMToolSelectorMiddleware

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A — TodoListMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Gives the agent a `write_todos` tool and a system-prompt extension that
    encourages it to break complex tasks into a checklist before executing.

WHY THIS MATTERS:
    Agents without task planning tend to lose track of multi-step goals,
    repeat steps, or stop before finishing. A to-do list keeps them on track
    and makes progress visible to humans watching the stream.

HOW IT WORKS:
    - Adds `write_todos` to the agent's tool list automatically
    - Agent calls it to record a plan: ["Step 1: ...", "Step 2: ..."]
    - Agent checks off items as it completes them
    - The todo list appears in agent state and can be read by middleware

WHEN TO USE:
    Multi-file code generation tasks
    Research + report writing workflows
    Any task with 3+ sequential steps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART B — LLMToolSelectorMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Before calling the main model, uses a FAST/CHEAP model to look at the
    current query and select only the RELEVANT tools to include.

WHY THIS MATTERS:
    With 20+ tools, the tool list fills up the context window and confuses
    the model. LLMToolSelectorMiddleware reduces the active tool set to
    only what's needed for this specific query, saving tokens and improving
    focus.

HOW IT WORKS:
    - A small model (e.g. gpt-4.1-mini) reads the user query
    - It uses structured output to pick which tools are relevant
    - Only those tools are passed to the main model's context
    - always_include: tools that are ALWAYS in scope regardless

WHEN TO USE:
    Agents with 10+ tools
    Reducing token costs in tool-heavy agents
    Improving model accuracy by reducing irrelevant options
"""

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, LLMToolSelectorMiddleware


# ---------------------------------------------------------------------------
# Tools — a realistic large set to demonstrate selector
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}: [file content here]"

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    return f"Written {len(content)} chars to {path}"

def run_tests(test_suite: str) -> str:
    """Run a test suite and return results."""
    return f"Tests in {test_suite}: 42 passed, 0 failed"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [...]"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"

def query_database(sql: str) -> str:
    """Run a SQL query against the database."""
    return f"Query result: [rows]"

def deploy_service(service: str, env: str) -> str:
    """Deploy a service to an environment."""
    return f"Deployed {service} to {env}"

def get_metrics(service: str) -> str:
    """Get performance metrics for a service."""
    return f"Metrics for {service}: CPU 45%, Memory 60%"

def create_ticket(title: str, description: str) -> str:
    """Create a bug/feature ticket."""
    return f"Ticket created: #{len(title)}"

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to a target language."""
    return f"Translation ({target_lang}): {text}"


# ──────────────────────────────────────────────────────────────────────────
# PART A: TodoListMiddleware
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A — TodoListMiddleware")
print("=" * 60)

agent_todo = create_agent(
    model="gpt-4.1",
    tools=[read_file, write_file, run_tests],
    middleware=[
        TodoListMiddleware(),   # adds write_todos tool + planning system prompt
    ],
)

result = agent_todo.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Read the config.py file, update the DB_HOST to 'prod-db.example.com', "
            "write the changes back, then run the test suite."
        ),
    }]
})

print(f"Final response:\n{result['messages'][-1].content[:400]}")

# The agent should have called write_todos first with a 3-step plan,
# then executed each step in order.


# ──────────────────────────────────────────────────────────────────────────
# PART B: LLMToolSelectorMiddleware
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B — LLMToolSelectorMiddleware (10 tools, picks 3)")
print("=" * 60)

all_tools = [
    read_file, write_file, run_tests,
    search_web, send_email, query_database,
    deploy_service, get_metrics, create_ticket, translate_text,
]

agent_selector = create_agent(
    model="gpt-4.1",
    tools=all_tools,                   # 10 tools — most irrelevant per query
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4.1-mini",      # cheap model does the selection
            max_tools=3,               # pass at most 3 tools to main model
            always_include=["search_web"],  # search is always available
        ),
    ],
)

# This query only needs: query_database (and always_included search_web)
result2 = agent_selector.invoke({
    "messages": [{
        "role": "user",
        "content": "Get me all users from the database who signed up last month.",
    }]
})
print(f"Response:\n{result2['messages'][-1].content[:400]}")
# Main model only sees: query_database + search_web (and maybe 1 more)
# NOT: deploy_service, send_email, translate_text, etc.


# ──────────────────────────────────────────────────────────────────────────
# PART C: Combined — TodoList + LLMToolSelector together
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART C — Combined: planning + smart tool selection")
print("=" * 60)

agent_combined = create_agent(
    model="gpt-4.1",
    tools=all_tools,
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4.1-mini",
            max_tools=4,
            always_include=["search_web"],
        ),
        TodoListMiddleware(),
    ],
)
print("Combined agent ready: plans tasks AND selects tools per step.")
print("Each step of the todo list gets only the relevant tools.")