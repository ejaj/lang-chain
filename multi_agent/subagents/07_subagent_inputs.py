"""
Controlling what context subagents receive
==================================================================

WHAT IS IT?
-----------
By default, you pass a query string to the subagent and that's it.
But sometimes the subagent needs more context to do a good job —
conversation history, results from previous agents, user preferences.

TWO OPTIONS:
  Option A — Query only   : fast, cheap, good for self-contained tasks
  Option B — Full context : better quality, costs more tokens

WHEN TO USE FULL CONTEXT:
- The subagent's task depends on what happened earlier
  e.g. a writer agent needs the research results from the previous step
- The subagent needs user preferences (tone, format, language)
- A previous agent's output should inform this agent's work
"""

from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.tools import ToolRuntime

# ==================================================================
# OPTION A — Query only (minimal, fast)
# ==================================================================
# WHEN TO USE: task is self-contained, doesn't depend on prior steps
# DOWNSIDE: subagent has no idea what the user said, what tone
#           they want, or what previous agents found

writer_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[],
    system_prompt="You write content."
)

@tool("writer_basic", description="Write content about a topic")
def call_writer_basic(topic: str):
    # Subagent only receives the topic string — nothing else
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": topic}]
    })
    return result["messages"][-1].content

# Usage:
# call_writer_basic("quantum computing")
# → writer has: topic only
# → doesn't know the user wanted a casual tone
# → doesn't know research findings from the previous step


# ==================================================================
# OPTION B — Full context (better quality)
# ==================================================================
# WHEN TO USE: task depends on prior steps or user preferences
# HOW: use ToolRuntime to access the main agent's state,
#      then build a richer prompt for the subagent

# Define custom state that holds extra data from previous steps
class CustomState(AgentState):
    research_results: str   # filled in by the research agent (07_outputs)
    user_tone: str          # e.g. "formal", "casual", "technical"
    target_audience: str    # e.g. "beginners", "executives"

@tool("writer_full", description="Write content about a topic using research context")
def call_writer_full(topic: str, runtime: ToolRuntime[None, CustomState]):
    # Pull rich context from the main agent's state
    research   = runtime.state.get("research_results", "No prior research")
    tone       = runtime.state.get("user_tone", "professional")
    audience   = runtime.state.get("target_audience", "general audience")

    # Build a richer prompt for the subagent
    prompt = f"""Write about: {topic}

Tone: {tone}
Target audience: {audience}

Use these research findings:
{research}

Make sure the content is accurate and matches the tone/audience.
"""
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return result["messages"][-1].content

# Usage:
# (After research agent has run and stored results in state)
# call_writer_full("quantum computing", runtime)
# → writer has: topic + tone + audience + research findings
# → produces much better, contextually appropriate output


# ==================================================================
# PUTTING IT TOGETHER — main agent that passes context to writer
# ==================================================================

@tool
def web_search():
    print("Web search")
research_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[web_search],
    system_prompt="You research topics and summarise findings clearly."
)

@tool("research", description="Research a topic and return findings")
def call_research(query: str, runtime: ToolRuntime[None, CustomState]):
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    findings = result["messages"][-1].content

    # Store findings in state so call_writer_full can use them
    # (requires Command pattern — see 08_subagent_outputs.py)
    return findings

main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[call_research, call_writer_full],
    system_prompt="You coordinate research and writing tasks.",
    state_schema=CustomState,
)

response = main_agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Write a casual blog post about quantum computing for beginners."
    }],
    "user_tone": "casual",
    "target_audience": "beginners",
    "research_results": "",  # will be filled by research agent
})

print(response["messages"][-1].content)

# WHAT HAPPENS:
# 1. Main agent sees: casual tone, beginner audience
# 2. Calls research agent → gets findings → stored in state
# 3. Calls writer agent → writer receives topic + tone + audience + findings
# 4. Writer produces a well-informed, casual, beginner-friendly post