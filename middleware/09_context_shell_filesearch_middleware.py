"""
TOPIC: ContextEditingMiddleware + ShellToolMiddleware + FilesystemFileSearchMiddleware

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A — ContextEditingMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Manages conversation context by clearing OLD tool call outputs when
    the total token count gets too large. Keeps only the N most recent
    tool results visible to the model.

WHY THIS MATTERS:
    Tool outputs can be huge (e.g. database dumps, web pages). After 10+
    tool calls, the context window fills with stale results. ContextEditing
    trims them while keeping recent results intact.

CONFIGURATION:
    ClearToolUsesEdit(
        trigger=100000,   # start trimming when context hits 100K tokens
        keep=3,           # always keep the last 3 tool call outputs
    )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART B — ShellToolMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Gives the agent a persistent bash shell session. The agent can run
    any shell command and the session state (env vars, working directory)
    persists between calls.

SECURITY POLICIES:
    HostExecutionPolicy         → run directly on the host OS  (dev only)
    DockerExecutionPolicy       → run inside a Docker container (recommended)
    CodexSandboxExecutionPolicy → run in a sandboxed environment

⚠️  NOTE: Does not currently work with HumanInTheLoopMiddleware.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART C — FilesystemFileSearchMiddleware
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT DOES:
    Adds two tools to the agent:
      glob_search → find files by name patterns  (e.g. "*.py", "**/*.json")
      grep_search → search file content by regex (e.g. "def create_agent")

WHEN TO USE:
    Code exploration agents
    Agents that need to find files before reading them
    Large codebases where the agent must discover structure first
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ShellToolMiddleware,
    HostExecutionPolicy,
    FilesystemFileSearchMiddleware,
)


# ---------------------------------------------------------------------------
# PART A: ContextEditingMiddleware
# ---------------------------------------------------------------------------
def big_database_query(sql: str) -> str:
    """Run a database query — returns large output."""
    # Simulate large output
    rows = [f"row_{i}: {{id: {i}, data: 'value_{i}'}}" for i in range(100)]
    return "\n".join(rows)


def summarize_data(data: str) -> str:
    """Summarize data."""
    return f"Summary: {len(data.split())} words processed."


print("=" * 60)
print("PART A — ContextEditingMiddleware")
print("=" * 60)

agent_context = create_agent(
    model="gpt-4.1",
    tools=[big_database_query, summarize_data],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100_000,   # tokens — start clearing when context is large
                    keep=3,            # always keep last 3 tool results
                ),
            ],
        ),
    ],
)

result = agent_context.invoke({
    "messages": [{
        "role": "user",
        "content": "Query the DB 5 times with different queries, then summarize all results.",
    }]
})
print(f"Response: {result['messages'][-1].content[:300]}")
print("(Old tool results were cleared from context once token limit was approached)")
print()


# ---------------------------------------------------------------------------
# PART B: ShellToolMiddleware
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART B — ShellToolMiddleware")
print("=" * 60)

agent_shell = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/tmp/agent_workspace",   # agent's working directory
            execution_policy=HostExecutionPolicy(),  # run on host (dev only!)
            # For production use:
            # execution_policy=DockerExecutionPolicy(image="python:3.11-slim")
        ),
    ],
)

result = agent_shell.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Create a file called hello.txt with 'Hello World' in it, "
            "then list the directory contents."
        ),
    }]
})
print(f"Response: {result['messages'][-1].content[:400]}")
print()

# The agent will call the shell tool:
#   shell("echo 'Hello World' > /tmp/agent_workspace/hello.txt")
#   shell("ls /tmp/agent_workspace/")
# Both run in the SAME persistent session.


# ---------------------------------------------------------------------------
# PART C: FilesystemFileSearchMiddleware
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART C — FilesystemFileSearchMiddleware")
print("=" * 60)

agent_search = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",   # only search within this directory
            use_ripgrep=True,         # use ripgrep for faster grep (if installed)
        ),
    ],
)

result = agent_search.invoke({
    "messages": [{
        "role": "user",
        "content": (
            "Find all Python files in the project, "
            "then search for any file that contains 'create_agent'."
        ),
    }]
})
print(f"Response: {result['messages'][-1].content[:400]}")
# Agent uses:
#   glob_search(pattern="**/*.py")
#   grep_search(pattern="create_agent", path="/workspace")
print()


# ---------------------------------------------------------------------------
# PART D: Combined — shell + file search (developer agent pattern)
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART D — Developer agent: shell + file search combined")
print("=" * 60)

dev_agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(root_path="/workspace", use_ripgrep=True),
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)
print("Developer agent ready:")
print("  • glob_search / grep_search for exploring the codebase")
print("  • Persistent shell for running commands, tests, scripts")