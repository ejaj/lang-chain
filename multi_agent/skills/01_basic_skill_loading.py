"""
Core skills pattern
================================================

WHAT IS IT?
-----------
The simplest form of the skills pattern. The agent has one tool:
load_skill(skill_name). When it needs expertise, it calls this tool,
gets back a specialized prompt, and uses that knowledge to respond.

Skills are stored as plain strings (prompts). In production these
would live in files or a database so teams can edit them independently.

WHEN TO USE:
- Agent needs to switch between knowledge domains
- You want to avoid one huge system prompt with all knowledge
- Different teams maintain different specializations
- Context window is limited and you can't load everything upfront

WHY PROGRESSIVE DISCLOSURE:
Loading all skills upfront = huge context window, higher cost,
model may get confused by unrelated domain knowledge.
Loading on demand = small context, only relevant knowledge loaded,
cheaper and more focused responses.

SCENARIO: A general assistant that can write SQL or review legal docs.

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 01_basic_skill_loading.py
"""

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------------
# Skill definitions — in production, load from files or a database
# ------------------------------------------------------------------
# Each skill is a specialized prompt. Teams own their own skill strings.
# The agent's system prompt stays small — skills load only when needed.

SKILLS: dict[str, str] = {
    "write_sql": """
You are now in SQL EXPERT mode.

Your job is to write correct, efficient SQL queries.

Guidelines:
- Always use explicit JOIN syntax (not implicit comma joins)
- Use CTEs (WITH clauses) for complex queries to improve readability
- Add comments explaining non-obvious logic
- Consider indexes and performance for large tables
- Validate edge cases: NULLs, empty sets, duplicates

Common schema patterns to know:
- customers(id, name, email, created_at, region)
- orders(id, customer_id, amount, status, created_at)
- order_items(id, order_id, product_id, quantity, unit_price)
- products(id, name, category, price, stock_qty)

When writing a query:
1. Clarify what the user wants if ambiguous
2. State any assumptions you're making
3. Provide the query with inline comments
4. Explain what the query does in plain English
""",

    "review_legal_doc": """
You are now in LEGAL DOCUMENT REVIEW mode.

Your job is to review contracts and legal documents for risks and issues.

Always check for:
1. LIABILITY GAPS     — Is liability capped? Is the cap reasonable?
2. TERMINATION CLAUSES — How can each party exit? What notice is required?
3. PAYMENT TERMS      — When is payment due? What are the penalties for late payment?
4. IP OWNERSHIP       — Who owns work product? Are there assignment clauses?
5. DISPUTE RESOLUTION — Arbitration or court? Which jurisdiction? Which law governs?
6. MISSING CLAUSES    — Force majeure, confidentiality, non-compete, indemnification?

Format your review as:
RISK LEVEL: low / medium / high
KEY FINDINGS:
- [finding 1]
- [finding 2]
RECOMMENDATION: approve / revise / reject
PRIORITY CHANGES NEEDED:
- [change 1]
- [change 2]

IMPORTANT: This is not legal advice. Always recommend the user
consult a qualified lawyer before signing.
""",

    "write_python": """
You are now in PYTHON EXPERT mode.

Your job is to write clean, production-quality Python code.

Standards:
- Follow PEP 8 style guidelines
- Use type hints on all function signatures
- Write docstrings for all public functions and classes
- Handle errors explicitly — no bare except clauses
- Prefer composition over inheritance
- Write testable code (pure functions where possible)

When writing code:
1. Confirm requirements before coding if anything is ambiguous
2. State what libraries/Python version you're assuming
3. Include example usage in a `if __name__ == "__main__":` block
4. Add inline comments for non-obvious logic
5. Mention any edge cases the caller should handle
""",
}

# ------------------------------------------------------------------
# The skill loading tool
# ------------------------------------------------------------------

@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill to gain domain expertise.

    Available skills:
    - write_sql        : Expert SQL query writing with schema knowledge
    - review_legal_doc : Legal document risk assessment and review
    - write_python     : Production-quality Python code writing

    Call this before attempting tasks in these domains.
    The skill will give you specialized knowledge and guidelines.
    """
    skill = SKILLS.get(skill_name)
    if not skill:
        available = ", ".join(SKILLS.keys())
        return f"Skill '{skill_name}' not found. Available skills: {available}"
    return skill.strip()

# ------------------------------------------------------------------
# Agent — starts lean, loads skills on demand
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

agent = create_react_agent(
    model=model,
    tools=[load_skill],
    prompt=(
        "You are a helpful technical assistant. "
        "You have access to specialized skills via the load_skill tool. "
        "Available skills: write_sql, review_legal_doc, write_python. "
        "Always load the relevant skill before attempting specialized tasks. "
        "Do not attempt SQL, legal review, or Python tasks without loading the skill first."
    )
)

# ------------------------------------------------------------------
# Helper: single-turn chat
# ------------------------------------------------------------------

def chat(history: list, user_message: str) -> tuple[str, list]:
    history.append(HumanMessage(content=user_message))
    result = agent.invoke({"messages": history})
    reply = result["messages"][-1].content
    history = result["messages"]
    return reply, history

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    history = []

    print("=== Skills Demo: Basic Skill Loading ===\n")

    # Turn 1: SQL task — agent loads write_sql skill
    print("USER: Write me a SQL query to find the top 5 customers by total order value this year.")
    reply, history = chat(history, "Write me a SQL query to find the top 5 customers by total order value this year.")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 2: Legal task — agent loads review_legal_doc skill
    contract = """
    SERVICE AGREEMENT between Acme Corp and Dev Studio.
    Term: 12 months. Payment: $10,000/month net-30.
    Liability: capped at one month fees.
    Termination: 30 days notice by either party.
    Governing law: not specified.
    """
    print("USER: Can you review this contract for risks?")
    reply, history = chat(history, f"Can you review this contract for risks?\n\n{contract}")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 3: Python task — agent loads write_python skill
    print("USER: Write a Python function that retries a failed HTTP request up to 3 times.")
    reply, history = chat(history, "Write a Python function that retries a failed HTTP request up to 3 times.")
    print(f"AGENT: {reply}\n")