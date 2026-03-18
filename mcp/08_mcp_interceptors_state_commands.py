"""
TOPIC: MCP Interceptors — State Updates and Commands

WHAT THIS COVERS:
    Interceptors can return Command objects instead of just tool results.
    This lets them update agent state AND control the graph execution flow
    based on what a tool did.

TWO COMMAND CAPABILITIES:
    1. Command(update={...})              → update agent state fields
    2. Command(update={...}, goto="node") → update state AND jump to a node

COMMON PATTERNS:
    Mark task complete when a specific tool succeeds
    Hand off to another agent after a tool runs
    End execution early when a task is done
    Track progress across multiple tool calls

WHY THIS MATTERS:
    Without Commands, tool results only appear in the conversation as messages.
    With Commands, tools can drive the agent's control flow and state.
"""

import asyncio
from langchain.agents import AgentState, create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1: Update state when a tool completes
# Mark a task as complete in agent state after a specific tool runs.
# ─────────────────────────────────────────────────────────────────────────────

async def mark_task_complete(
    request: MCPToolCallRequest,
    handler,
):
    """
    After submit_order runs successfully, mark task_status as 'completed'.
    Future middleware or tools can read this state to know the task is done.
    """
    result = await handler(request)

    if request.name == "submit_order":
        print(f"  [interceptor] submit_order completed → marking task complete")
        return Command(
            update={
                "messages":    [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",   # written to agent state permanently
            }
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2: Hand off to another agent/node after a tool runs
# Use goto to jump to a different node after the tool completes.
# ─────────────────────────────────────────────────────────────────────────────

async def handoff_to_summary_agent(
    request: MCPToolCallRequest,
    handler,
):
    """
    After submit_order, update state and jump to the summary_agent node.
    The summary_agent will then handle the final response to the user.
    """
    result = await handler(request)

    if request.name == "submit_order":
        print(f"  [interceptor] submit_order done → handing off to summary_agent")
        return Command(
            update={
                "messages":    [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",
                "order_id":    "order-xyz",   # pass data to the next agent
            },
            goto="summary_agent",   # jump to this node in the graph
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3: End execution early on success
# Use goto="__end__" to stop the agent run immediately.
# ─────────────────────────────────────────────────────────────────────────────

async def end_on_completion(
    request: MCPToolCallRequest,
    handler,
):
    """
    When mark_complete tool runs, end the agent run immediately.
    No more model calls — the task is done.
    """
    result = await handler(request)

    if request.name == "mark_complete":
        print(f"  [interceptor] Task marked complete → ending agent run")
        return Command(
            update={
                "messages": [result] if isinstance(result, ToolMessage) else [],
                "status":   "done",
            },
            goto="__end__",   # end the agent run
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4: Track progress across multiple tool calls
# Increment a counter in state each time a specific tool runs.
# ─────────────────────────────────────────────────────────────────────────────

async def track_progress(
    request: MCPToolCallRequest,
    handler,
):
    """
    Track how many steps have been processed.
    Reads current count from state, increments it, writes back via Command.
    """
    result = await handler(request)

    if request.name == "process_step":
        runtime       = request.runtime
        current_count = runtime.state.get("steps_completed", 0)
        new_count     = current_count + 1

        print(f"  [interceptor] Step {new_count} completed")

        return Command(
            update={
                "messages":        [result] if isinstance(result, ToolMessage) else [],
                "steps_completed": new_count,
            }
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED EXAMPLE
# Chain multiple command-returning interceptors together.
# ─────────────────────────────────────────────────────────────────────────────

async def combined_example():
    """Combine progress tracking and completion detection."""
    client = MultiServerMCPClient(
        {
            "workflow": {
                "transport": "http",
                "url":       "http://localhost:8001/mcp",
            }
        },
        tool_interceptors=[
            track_progress,      # runs first (outermost)
            end_on_completion,   # runs second (innermost)
        ],
    )

    tools = await client.get_tools()
    agent = create_agent("gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages":        [{"role": "user", "content": "Process all steps then mark complete"}],
        "steps_completed": 0,
        "status":          "in_progress",
    })

    print(f"steps_completed : {result.get('steps_completed')}")
    print(f"status          : {result.get('status')}")
    print(f"Response        : {result['messages'][-1].content[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print("""
Interceptor Command Quick Reference:

  Update state only:
    return Command(
        update={"field": value, "messages": [result]},
    )

  Update state AND jump to a node:
    return Command(
        update={"field": value, "messages": [result]},
        goto="node_name",
    )

  End the agent run:
    return Command(
        update={"status": "done", "messages": [result]},
        goto="__end__",
    )

  Important:
    Always include "messages": [result] in update if result is a ToolMessage,
    so the tool result still appears in the conversation history.

  Command vs plain return:
    return result         → only adds tool message to conversation
    return Command(...)   → adds tool message + updates state + optionally redirects
""")

if __name__ == "__main__":
    asyncio.run(combined_example())