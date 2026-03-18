"""
TOPIC: MCP Progress Notifications + Server Logging

WHAT THIS COVERS:
    MCP servers can push events back to the client WHILE a tool is running:
    1. Progress notifications → "30% done", "processing file 3 of 10"
    2. Log messages           → server-side logs forwarded to your client

WHY THIS MATTERS:
    Long-running tools (file processing, web scraping, ML inference)
    take time. Without progress events, the user just waits with no feedback.
    With callbacks, you can show a live progress bar or status updates.

HOW IT WORKS:
    Pass a Callbacks object to MultiServerMCPClient.
    Two callbacks:
        on_progress        → fires when the server reports progress
        on_logging_message → fires when the server emits a log message

CALLBACK CONTEXT (CallbackContext):
    context.server_name → which MCP server sent the event
    context.tool_name   → which tool is running (None for log messages outside tools)
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

async def on_progress(
    progress: float,
    total:    float | None,
    message:  str | None,
    context:  CallbackContext,
):
    """
    Called by the client whenever the MCP server sends a progress notification.

    Args:
        progress → current progress value (e.g. 30.0)
        total    → total value (e.g. 100.0) — None if unknown
        message  → human-readable status message from the server
        context  → server_name and tool_name
    """
    server = context.server_name
    tool   = context.tool_name or "unknown"

    if total:
        pct    = progress / total * 100
        filled = int(pct / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        print(f"  [{server}/{tool}] [{bar}] {pct:.0f}% — {message}")
    else:
        # Total unknown — just show raw progress
        print(f"  [{server}/{tool}] Progress: {progress} — {message}")


async def progress_example():
    """Agent with live progress updates from the MCP server."""
    client = MultiServerMCPClient(
        {
            "processor": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        callbacks=Callbacks(on_progress=on_progress),   # attach progress callback
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    print("Starting long-running tool — watch for progress events:")
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Process all 50 records in the dataset"}]
    })
    print(f"\nFinal: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# SERVER LOGGING
# ─────────────────────────────────────────────────────────────────────────────

try:
    from mcp.types import LoggingMessageNotificationParams
except ImportError:
    LoggingMessageNotificationParams = None


async def on_logging_message(
    params:  "LoggingMessageNotificationParams",
    context: CallbackContext,
):
    """
    Called when the MCP server emits a log message.
    Useful for debugging server-side issues from the client.

    params.level → "debug" | "info" | "warning" | "error" | "critical"
    params.data  → the log message content
    """
    server = context.server_name
    level  = params.level.upper()

    # Route to appropriate output based on log level
    if params.level in ("error", "critical"):
        print(f" [{server}] {level}: {params.data}")
    elif params.level == "warning":
        print(f"[{server}] {level}: {params.data}")
    else:
        print(f"[{server}] {level}: {params.data}")


async def logging_example():
    """Agent that surfaces MCP server logs to the client."""
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      ["/path/to/math_server.py"],
            }
        },
        callbacks=Callbacks(on_logging_message=on_logging_message),   # attach log callback
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 100 divided by 4?"}]
    })
    print(f"Result: {result['messages'][-1].content}")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED — progress + logging at the same time
# ─────────────────────────────────────────────────────────────────────────────

async def combined_callbacks_example():
    """Attach both progress and logging callbacks to the same client."""
    client = MultiServerMCPClient(
        {
            "processor": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        callbacks=Callbacks(
            on_progress=on_progress,
            on_logging_message=on_logging_message,
        ),
    )

    tools  = await client.get_tools()
    agent  = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Process the large dataset"}]
    })
    print(f"Final: {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Progress + Logging Quick Reference:

  Progress callback:
    async def on_progress(progress, total, message, context):
        pct = progress / total * 100 if total else progress
        print(f"[{context.server_name}/{context.tool_name}] {pct:.0f}% — {message}")

  Logging callback:
    async def on_logging_message(params, context):
        print(f"[{context.server_name}] {params.level}: {params.data}")

  Attach to client:
    client = MultiServerMCPClient(
        {...},
        callbacks=Callbacks(
            on_progress=on_progress,
            on_logging_message=on_logging_message,
        ),
    )

  CallbackContext fields:
    context.server_name → which MCP server sent the event
    context.tool_name   → which tool is running (may be None)

  Log levels: "debug" | "info" | "warning" | "error" | "critical"
""")

if __name__ == "__main__":
    asyncio.run(combined_callbacks_example())