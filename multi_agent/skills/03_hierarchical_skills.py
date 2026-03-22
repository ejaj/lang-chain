"""
Hierarchical skills (tree structure)
=================================================================

WHAT IS IT?
-----------
Skills can contain OTHER SKILLS in a tree structure.
Loading a parent skill reveals what child skills exist.
The agent then loads child skills only when it needs them.

Think of it like a folder structure:
  data_science/
    ├── pandas_expert
    ├── visualization
    └── statistical_analysis

Loading "data_science" tells the agent these sub-skills exist.
The agent then loads "pandas_expert" only when asked to wrangle data,
"visualization" only when asked to make charts, etc.

WHY HIERARCHICAL:
- You might have 50+ skills total — listing all of them in the
  system prompt would be overwhelming
- Group related skills under a parent, load the parent first
  (progressive disclosure level 1), then load children as needed
  (progressive disclosure level 2)
- Each team can own a subtree independently

WHEN TO USE:
- Large skill libraries (10+ skills)
- Skills naturally group into domains
- You want coarse → fine progressive disclosure
- Teams own entire skill domains, not just individual skills

SCENARIO: A data science assistant
  Parent: "data_science" → lists available sub-skills
  Child:  "pandas_expert" → loaded when data wrangling is needed
  Child:  "visualization" → loaded when charting is needed
  Child:  "statistical_analysis" → loaded when stats are needed

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 03_hierarchical_skills.py
"""

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------------
# Hierarchical skill registry
# ------------------------------------------------------------------
# Skills can have:
#   "prompt"     → the specialized context to inject
#   "children"   → sub-skills available after this one is loaded
#   "parent"     → which skill this belongs to

SKILL_TREE: dict[str, dict] = {

    # ── Top-level parent skills ──────────────────────────────────
    "data_science": {
        "prompt": """
DATA SCIENCE SKILL LOADED.

You are now in data science mode. You can help with data analysis,
statistics, visualization, and machine learning.

Available sub-skills (load these for deeper expertise):
- pandas_expert        : DataFrame manipulation, cleaning, transformation
- visualization        : Charts, plots, dashboards with matplotlib/plotly
- statistical_analysis : Hypothesis testing, regression, distributions

Load a sub-skill when the user's question requires that specialization.
""",
        "children": ["pandas_expert", "visualization", "statistical_analysis"],
        "parent": None,
    },

    "software_engineering": {
        "prompt": """
SOFTWARE ENGINEERING SKILL LOADED.

You can help with system design, code review, and engineering practices.

Available sub-skills:
- python_expert  : Production Python, type hints, testing, packaging
- api_design     : REST/GraphQL design, OpenAPI, versioning
- code_review    : Security, performance, style, architecture review

Load a sub-skill for deeper expertise in that area.
""",
        "children": ["python_expert", "api_design", "code_review"],
        "parent": None,
    },

    # ── data_science children ─────────────────────────────────────
    "pandas_expert": {
        "prompt": """
PANDAS EXPERT SKILL LOADED.

Deep expertise in pandas DataFrame operations.

Key patterns:
- Use .loc[] for label-based, .iloc[] for position-based indexing
- Prefer vectorized operations over loops (.apply() is a last resort)
- Use pd.read_csv(dtype=...) to avoid dtype inference issues
- For large datasets: use chunking or consider polars/dask
- Always check .dtypes and .isnull().sum() before analysis

Common operations:
  # Groupby + agg
  df.groupby('category').agg({'sales': 'sum', 'qty': 'mean'})

  # Merge
  pd.merge(left, right, on='id', how='left')

  # Pivot
  df.pivot_table(values='sales', index='region', columns='month', aggfunc='sum')
""",
        "children": [],
        "parent": "data_science",
    },

    "visualization": {
        "prompt": """
VISUALIZATION SKILL LOADED.

Expert in matplotlib, seaborn, and plotly.

Chart selection guide:
- Comparison over time  → line chart
- Part-to-whole         → pie or stacked bar
- Distribution          → histogram or box plot
- Correlation           → scatter plot with regression line
- Category comparison   → bar chart (horizontal for long labels)

Always:
1. Label axes with units
2. Add a descriptive title
3. Include a legend when using multiple series
4. Use colorblind-friendly palettes (e.g. seaborn's 'colorblind')
""",
        "children": [],
        "parent": "data_science",
    },

    "statistical_analysis": {
        "prompt": """
STATISTICAL ANALYSIS SKILL LOADED.

Expert in applied statistics for data science.

Choosing the right test:
- Compare two groups (normal) → t-test
- Compare two groups (non-normal) → Mann-Whitney U
- Compare 3+ groups → ANOVA then post-hoc Tukey
- Correlation → Pearson (normal) or Spearman (non-normal)
- Categorical association → Chi-square test

Always:
1. State your null and alternative hypothesis
2. Check assumptions (normality, equal variance) before the test
3. Report effect size alongside p-value (p < 0.05 alone is not enough)
4. Correct for multiple comparisons (Bonferroni or FDR)
""",
        "children": [],
        "parent": "data_science",
    },

    # ── software_engineering children ────────────────────────────
    "python_expert": {
        "prompt": """
PYTHON EXPERT SKILL LOADED.

Production Python best practices.

Always:
- Use type hints on all function signatures
- Write docstrings (Google style)
- Use dataclasses or Pydantic for data structures
- Handle exceptions specifically — never bare except
- Use pathlib.Path instead of os.path
- Prefer f-strings over .format() or %

Testing:
- pytest for all tests
- Use fixtures for repeated setup
- Aim for >80% coverage on business logic
- Use hypothesis for property-based testing on complex inputs
""",
        "children": [],
        "parent": "software_engineering",
    },

    "api_design": {
        "prompt": """
API DESIGN SKILL LOADED.

REST and GraphQL design expert.

REST principles:
- Use nouns for resources: /users, /orders (not /getUsers)
- Use HTTP verbs correctly: GET (read), POST (create), PUT/PATCH (update), DELETE
- Return 201 for created, 204 for deleted, 422 for validation errors
- Version in URL: /v1/users (not headers — harder to test)
- Paginate list endpoints: ?page=1&per_page=20

Always provide:
- OpenAPI/Swagger spec
- Example request + response
- Error response schema with a machine-readable error code
""",
        "children": [],
        "parent": "software_engineering",
    },

    "code_review": {
        "prompt": """
CODE REVIEW SKILL LOADED.

Expert code reviewer focused on: security, performance, maintainability.

Always check:
SECURITY: SQL injection, XSS, hardcoded secrets, insecure deserialization
PERFORMANCE: N+1 queries, missing indexes, unnecessary loops, memory leaks
MAINTAINABILITY: function length (<20 lines), naming clarity, duplication
CORRECTNESS: edge cases (empty input, None, overflow), error handling

Format feedback as:
SEVERITY: critical / major / minor / suggestion
LOCATION: file:line
ISSUE: what's wrong
FIX: what to do instead
""",
        "children": [],
        "parent": "software_engineering",
    },
}

# ------------------------------------------------------------------
# Tool: load a skill (parent or child)
# ------------------------------------------------------------------

@tool
def load_skill(skill_name: str) -> str:
    """Load a skill to gain specialized expertise.

    Top-level skills (load these first to discover sub-skills):
    - data_science         : Data analysis, stats, visualization
    - software_engineering : Python, API design, code review

    After loading a parent skill, its sub-skills will be listed.
    Load sub-skills for deeper expertise in a specific area.
    """
    skill = SKILL_TREE.get(skill_name)
    if not skill:
        available = [k for k, v in SKILL_TREE.items() if v["parent"] is None]
        return f"Skill '{skill_name}' not found. Top-level skills: {available}"

    return skill["prompt"].strip()

# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

agent = create_react_agent(
    model=model,
    tools=[load_skill],
    prompt=(
        "You are a technical assistant with access to hierarchical skills. "
        "Top-level skills: data_science, software_engineering. "
        "Load a top-level skill first to discover its sub-skills, "
        "then load sub-skills for deeper expertise. "
        "Always load the most specific skill relevant to the task."
    )
)

# ------------------------------------------------------------------
# Helper
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

    print("=== Hierarchical Skills Demo ===\n")

    # Turn 1: data science question → loads data_science → finds pandas_expert
    print("USER: I have a DataFrame with missing values and duplicates. How do I clean it?")
    reply, history = chat(history, "I have a DataFrame with missing values and duplicates. How do I clean it?")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 2: visualization question → loads visualization sub-skill
    print("USER: Now I want to plot the distribution of values after cleaning.")
    reply, history = chat(history, "Now I want to plot the distribution of values after cleaning.")
    print(f"AGENT: {reply}\n")
    print("-" * 60)

    # Turn 3: switches to engineering domain → loads software_engineering
    print("USER: Can you review this Python function for issues?")
    code = """
def get_users(db, filter):
    query = "SELECT * FROM users WHERE name = '" + filter + "'"
    result = db.execute(query)
    return result
"""
    reply, history = chat(history, f"Can you review this Python function for issues?\n{code}")
    print(f"AGENT: {reply}\n")