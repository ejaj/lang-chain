"""
Loops and retry logic
==============================================

WHAT IS IT?
-----------
A conditional edge points BACK to an earlier node, creating a loop.
The loop continues until a quality condition is met or a max
iteration count is reached.

This pattern is also called "reflection" — the agent generates output,
a critic evaluates it, and if it's not good enough the agent tries again
with feedback.

WHEN TO USE:
- Output quality is variable and needs iterative refinement
- You want the model to self-correct based on a quality check
- You need retry logic on failures (API errors, bad output format)
- Iterative processes: draft → review → revise → review → ...

IMPORTANT — ALWAYS SET A MAX ITERATIONS LIMIT:
  Without a limit, a bug in your quality check can loop forever.
  Always track iteration count and break out after N attempts.

SCENARIO: A code generation loop with quality checking.
  Step 1 — generate : write Python code for the task
  Step 2 — evaluate : check if the code is correct and complete
    If PASS  → done, return the code
    If FAIL  → send feedback back to generate, try again (max 3 times)
  Step 3 — done     : return final code to user

INSTALL:
    pip install langchain-anthropic langgraph

RUN:
    export ANTHROPIC_API_KEY=your_key
    python 03_loops_and_retry.py
"""

from typing import TypedDict, Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------

class CodeState(TypedDict):
    task: str               # what code to write
    code: str               # current generated code
    feedback: str           # evaluator feedback (if any)
    iterations: int         # how many generate attempts so far
    max_iterations: int     # stop looping after this many attempts
    passed: bool            # whether the code passed evaluation
    final_code: str         # the accepted final code

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514")

def llm(system: str, user: str) -> str:
    response = model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()

# ------------------------------------------------------------------
# Node 1 — Generate
# ------------------------------------------------------------------

def generate_node(state: CodeState) -> dict:
    """Generate Python code. If there's feedback, incorporate it."""
    iteration = state["iterations"] + 1

    if state["feedback"]:
        # Re-generate with feedback from previous attempt
        prompt = (
            f"Task: {state['task']}\n\n"
            f"Your previous attempt had issues:\n{state['feedback']}\n\n"
            f"Previous code:\n```python\n{state['code']}\n```\n\n"
            f"Fix the issues and write improved code."
        )
        print(f"  [generate] attempt {iteration} (with feedback)")
    else:
        # First attempt
        prompt = f"Task: {state['task']}"
        print(f"  [generate] attempt {iteration} (first try)")

    code = llm(
        system=(
            "You are a Python expert. Write clean, working Python code.\n"
            "Include type hints, docstrings, and error handling.\n"
            "Return ONLY the code — no explanation, no markdown fences."
        ),
        user=prompt
    )

    return {"code": code, "iterations": iteration}

# ------------------------------------------------------------------
# Node 2 — Evaluate
# ------------------------------------------------------------------

def evaluate_node(state: CodeState) -> dict:
    """Check the generated code for correctness and completeness."""
    evaluation = llm(
        system=(
            "You are a strict code reviewer. Evaluate the code against the task.\n\n"
            "Check for:\n"
            "1. Does it solve the task correctly?\n"
            "2. Are there any syntax errors or obvious bugs?\n"
            "3. Does it handle edge cases (None, empty input, errors)?\n"
            "4. Are type hints and docstrings present?\n\n"
            "Start your response with PASS or FAIL.\n"
            "If FAIL, list specific issues that must be fixed."
        ),
        user=f"Task: {state['task']}\n\nCode:\n{state['code']}"
    )

    passed = evaluation.upper().startswith("PASS")
    feedback = "" if passed else evaluation

    print(f"  [evaluate] {'✓ PASS' if passed else '✗ FAIL'} — iteration {state['iterations']}")
    if not passed:
        print(f"    feedback: {feedback[:100]}...")

    return {"passed": passed, "feedback": feedback}

# ------------------------------------------------------------------
# Decision function — loop back or proceed to done
# ------------------------------------------------------------------

def should_continue(state: CodeState) -> Literal["generate", "done"]:
    """
    If code passed → done.
    If failed but under max_iterations → loop back to generate.
    If failed and at max_iterations → accept anyway (best effort).
    """
    if state["passed"]:
        return "done"
    if state["iterations"] >= state["max_iterations"]:
        print(f"  [loop] max iterations ({state['max_iterations']}) reached — accepting best attempt")
        return "done"
    return "generate"

# ------------------------------------------------------------------
# Node 3 — Done
# ------------------------------------------------------------------

def done_node(state: CodeState) -> dict:
    """Finalize and return the accepted code."""
    return {"final_code": state["code"]}

# ------------------------------------------------------------------
# Graph — generate → evaluate → (loop back OR done)
# ------------------------------------------------------------------

builder = StateGraph(CodeState)

builder.add_node("generate", generate_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("done",     done_node)

builder.add_edge(START,      "generate")
builder.add_edge("generate", "evaluate")

# Conditional edge: loop back to generate OR proceed to done
builder.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "generate": "generate",   # ← loop back
        "done":     "done",
    }
)

builder.add_edge("done", END)

graph = builder.compile()

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Loop and Retry Demo ===\n")

    tasks = [
        "Write a function that takes a list of integers and returns a new list "
        "with duplicates removed while preserving the original order.",

        "Write a function that safely reads a JSON file and returns a default "
        "value if the file doesn't exist or contains invalid JSON.",
    ]

    for task in tasks:
        print(f"TASK: {task[:80]}...")
        print("-" * 60)

        result = graph.invoke({
            "task":           task,
            "code":           "",
            "feedback":       "",
            "iterations":     0,
            "max_iterations": 3,
            "passed":         False,
            "final_code":     "",
        })

        print(f"\nCompleted in {result['iterations']} iteration(s)")
        print(f"Passed: {result['passed']}")
        print(f"\nFinal code:\n{result['final_code']}")
        print("\n" + "=" * 60 + "\n")