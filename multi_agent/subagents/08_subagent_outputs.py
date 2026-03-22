"""
Controlling what the main agent gets back
==================================================================

WHAT IS IT?
-----------
The main agent ONLY sees the subagent's last message.
It never sees the subagent's tool calls, reasoning, or intermediate steps.

COMMON FAILURE:
  Subagent does 10 tool calls → finds the answer → final message: "Done."
  Main agent receives "Done." → has nothing to work with → bad reply.

TWO STRATEGIES TO FIX THIS:
  Strategy A — Prompt the subagent to always end with a real summary
  Strategy B — Format the output in code + optionally pass back extra state

WHEN TO USE EACH:
  Strategy A : simple, always use this as a baseline
  Strategy B : when you also need to pass data between agents via state
               (e.g. research results that the writer agent will need)
"""

from typing import Annotated
from langchain.tools import tool, InjectedToolCallId
from langchain.agents import create_agent
from langchain.schema import ToolMessage
from langgraph.types import Command

# ==================================================================
# STRATEGY A — Prompt the subagent to summarise in its final message
# ==================================================================
# WHEN TO USE: always — this is the minimum you should do
# HOW: tell the subagent in its system prompt that the supervisor
#      only sees the final message, so it must be complete
@tool
def web_search():
    print("Web search")
@tool
def read_url():
    print("read url")
    
research_agent_a = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[web_search, read_url],
    system_prompt="""You are a research specialist.

IMPORTANT: Your supervisor only sees your FINAL message — not your
tool calls or reasoning steps. Always end with a complete, structured
summary of your findings. Never end with just "Done" or "Completed."

Always format your final message as:

FINDINGS:
[your research results here]

CONFIDENCE: high / medium / low
SOURCES: [list the URLs or sources you used]
"""
)

@tool("research_a", description="Research a topic and return structured findings")
def call_research_a(query: str):
    result = research_agent_a.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    # This will now always contain FINDINGS / CONFIDENCE / SOURCES
    return result["messages"][-1].content

# RESULT: main agent always gets a structured, usable summary


# ==================================================================
# STRATEGY B — Format in code + pass extra state back
# ==================================================================
# WHEN TO USE: when other agents downstream need this agent's results
#              (not just the main agent, but via shared state)
# HOW: return a Command object that updates both the message AND
#      named state keys that other tools can read

research_agent_b = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[web_search, read_url],
    system_prompt="You are a research specialist. Always end with a "
                  "complete summary of your findings."
)

@tool("research_b", description="Research a topic and store findings in state")
def call_research_b(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],  # injected automatically
) -> Command:
    result = research_agent_b.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    final_message = result["messages"][-1].content

    # Return a Command that updates BOTH the message AND named state keys
    return Command(update={
        # These state keys are now available to all subsequent tool calls
        # (see 07_subagent_inputs.py — writer_full reads research_results)
        "research_results": final_message,
        "research_sources": result.get("sources", []),

        # This is the message the main agent sees in its conversation
        "messages": [
            ToolMessage(
                content=f"Research complete.\n\n{final_message}",
                tool_call_id=tool_call_id
            )
        ]
    })

# RESULT:
# - Main agent sees: "Research complete. FINDINGS: ..."
# - State["research_results"] is now set
# - When writer agent runs (07b), it reads research_results from state
#   even though it never directly called research


# ==================================================================
# COMBINING BOTH STRATEGIES — recommended approach
# ==================================================================

research_agent_best = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[web_search, read_url],
    # Strategy A: prompt ensures good final message
    system_prompt="""You are a research specialist.

Your supervisor only sees your FINAL message. Always end with:
FINDINGS: [complete summary]
CONFIDENCE: high/medium/low
SOURCES: [list]
"""
)

@tool("research_best", description="Research a topic and return structured findings")
def call_research_best(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = research_agent_best.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    final_message = result["messages"][-1].content

    # Strategy B: also store in state for downstream agents
    return Command(update={
        "research_results": final_message,  # writer agent will use this
        "messages": [
            ToolMessage(
                content=final_message,
                tool_call_id=tool_call_id
            )
        ]
    })

# Strategy A guarantees the content is good.
# Strategy B makes it available to downstream agents via state.
# Use both together for maximum reliability.