"""
How to tell the main agent what agents exist
====================================================================

WHAT IS IT?
-----------
When using the single dispatch pattern (05), the main agent needs
to know which agents are available and what they do.
There are 3 ways to provide this information.

WHY IT MATTERS:
The main agent decides which agent to call based purely on names
and descriptions. Vague names → wrong routing. Clear names → correct.

  BAD:  "agent1", "agent2"       → main agent guesses randomly
  GOOD: "legal_reviewer", "crm"  → main agent routes correctly

THREE METHODS:
  Method A — System prompt   : list agents in the prompt (< 10 agents)
  Method B — Enum constraint : type-safe, schema-enforced names
  Method C — Tool discovery  : main agent calls list_agents() on demand
"""

from langchain.tools import tool
from langchain.agents import create_agent

# ==================================================================
# METHOD A — System prompt enumeration
# ==================================================================
# WHEN TO USE: < 10 agents, rarely changing list, want simplicity
# DOWNSIDE: changing agents = editing the system prompt manually

research_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[]
)

writer_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[]
)

reviewer_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[]
)



AGENTS_A = {
    "research": research_agent,
    "writer":   writer_agent,
    "reviewer": reviewer_agent,
}

@tool
def task_a(agent_name: str, description: str) -> str:
    """Run a task using a named specialist agent."""
    agent = AGENTS_A.get(agent_name)
    if not agent:
        return f"Unknown agent: {agent_name}"
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content

# Agent list is baked into the system prompt
main_agent_a = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[task_a],
    system_prompt=(
        "You coordinate specialist agents. Use the task tool.\n\n"
        "Available agents:\n"
        "- research: Web research and fact-finding. Use for factual "
        "questions, current events, technical topics.\n"
        "- writer: Content creation. Use when the user wants an article, "
        "email, report, or any written output.\n"
        "- reviewer: Quality checking. Use after writing to check accuracy "
        "and grammar before delivering to the user.\n"
    )
)


# ==================================================================
# METHOD B — Enum constraint
# ==================================================================
# WHEN TO USE: < 10 agents, want type safety, prefer schema validation
# BENEFIT: model CANNOT hallucinate an agent name not in the enum
# DOWNSIDE: adding an agent = code change (add to enum)

from enum import Enum

class AgentName(str, Enum):
    RESEARCH = "research"
    WRITER   = "writer"
    REVIEWER = "reviewer"
    # To add a new agent: add a line here + add to AGENTS_B dict

AGENTS_B = {
    AgentName.RESEARCH: research_agent,
    AgentName.WRITER:   writer_agent,
    AgentName.REVIEWER: reviewer_agent,
}

@tool
def task_b(agent_name: AgentName, description: str) -> str:
    """Run a task using a named specialist agent."""
    # agent_name is validated against the enum by the framework
    # If the model tries "legal" and it's not in AgentName → error
    agent = AGENTS_B[agent_name]
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content

main_agent_b = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[task_b],
    system_prompt="You coordinate specialist agents. Use the task tool."
    # No need to list agents in prompt — the enum schema tells the model
)


# ==================================================================
# METHOD C — Tool-based discovery
# ==================================================================
# WHEN TO USE: 10+ agents, dynamic/growing registry, multi-team setup
# BENEFIT: system prompt stays tiny, new agents auto-discoverable
# DOWNSIDE: main agent must call list_agents before calling task
#           (costs an extra LLM turn)

legal_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[]
)

finance_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[]
)



AGENTS_C = {
    "research":  {"agent": research_agent,  "description": "Web research and fact-finding"},
    "writer":    {"agent": writer_agent,    "description": "Content creation and editing"},
    "reviewer":  {"agent": reviewer_agent,  "description": "Document and code review"},
    "legal":     {"agent": legal_agent,     "description": "Legal questions and contracts"},
    "financial": {"agent": finance_agent,   "description": "Financial analysis and modelling"},
    # Team adds their agent here — no other code changes
}

@tool
def list_agents(query: str = "") -> str:
    """List available specialist agents, optionally filtered by topic.

    Call this first to discover what agents are available,
    then use the task tool to invoke the right one.
    """
    matches = {
        name: info["description"]
        for name, info in AGENTS_C.items()
        if not query or query.lower() in info["description"].lower()
    }
    if not matches:
        return f"No agents found matching '{query}'"
    return "\n".join(f"- {name}: {desc}" for name, desc in matches.items())

@tool
def task_c(agent_name: str, description: str) -> str:
    """Run a task using a named specialist agent."""
    info = AGENTS_C.get(agent_name)
    if not info:
        return f"Unknown agent '{agent_name}'. Use list_agents to see available agents."
    result = info["agent"].invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content

# System prompt is tiny — no hardcoded agent list
main_agent_c = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[list_agents, task_c],
    system_prompt=(
        "You coordinate specialist agents. "
        "First use list_agents to discover available agents, "
        "then use task to invoke the right one."
    )
)

# FLOW WITH METHOD C:
# 1. User: "Find me information about semiconductor export laws"
# 2. Main agent calls: list_agents("legal")
#    → "- legal: Legal questions and contracts"
# 3. Main agent calls: task("legal", "find semiconductor export restrictions 2025")
#    → legal agent researches and returns findings
# 4. Main agent replies to user