import time 
"""
Asynchronous subagent execution
========================================================

WHAT IS IT?
-----------
Main agent kicks off a background job and stays responsive.
The user can keep chatting while the subagent works in the background.
Uses a 3-tool pattern: start_job / check_status / get_result.

NOTE: "async" here does NOT mean Python's async/await.
It means the subagent runs in a separate background process
(e.g. a Celery task, a thread, a separate service).

WHEN TO USE:
- Long-running tasks that take minutes or hours
- The subagent's work is independent of the conversation flow
- Users should not have to wait staring at a frozen screen
- You want to run multiple independent tasks in parallel

WHY USE ASYNC:
- Reviewing a 150-page contract takes 5 minutes.
  With sync → chat freezes for 5 minutes.
  With async → user gets a job ID in 1 second, can keep chatting.

WATCH OUT:
- More complex to implement than sync
- Requires a job storage system (Redis, database, etc.)
- You need a way to notify the user when the job finishes

FLOW:
    User submits task
      ↓
    Main agent calls start_job() → returns job_id immediately
      ↓
    User keeps chatting (main agent still responsive)
      ↓  (background)
    Subagent works...
      ↓
    User asks "is it done?" → main agent calls check_status(job_id)
      ↓
    When done → main agent calls get_result(job_id) → replies
"""

import uuid
import threading
from langchain.tools import tool
from langchain.agents import create_agent

# ------------------------------------------------------------------
# Job store (use Redis or a database in production)
# ------------------------------------------------------------------

jobs = {}  # { job_id: { "status": "running"|"completed"|"failed", "result": str } }

# ------------------------------------------------------------------
# The actual subagent that runs in the background
# ------------------------------------------------------------------

@tool
def read_document(text: str) -> str:
    """Read and extract key clauses from a legal document."""
    time.sleep(1)  # simulate document parsing
    return (
        "Document parsed. Sections found: Parties, Term (12 months), "
        "Payment ($10,000/month, net-30), Liability cap (1 month fees), "
        "Termination (30 days notice), Governing law (not specified)."
    )
 
@tool
def search_case_law(query: str) -> str:
    """Search relevant case law for a legal topic."""
    time.sleep(1)  # simulate API call
    return (
        f"Relevant cases for '{query}': "
        "Smith v Jones (2021) — liability caps upheld when clearly stated. "
        "ABC Corp v XYZ Ltd (2023) — termination without cause requires 60 days in SaaS contracts."
    )

legal_reviewer = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[read_document, search_case_law],
    system_prompt="You are a legal document reviewer. Analyse contracts "
                  "thoroughly and provide a detailed risk assessment."
)

def run_review_in_background(job_id: str, document: str):
    """Runs in a background thread. Updates jobs dict when done."""
    try:
        result = legal_reviewer.invoke({
            "messages": [{"role": "user", "content": f"Review this contract:\n\n{document}"}]
        })
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result["messages"][-1].content
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(e)

# ------------------------------------------------------------------
# 3 tools the main agent uses to manage async work
# ------------------------------------------------------------------

@tool
def start_review(document: str) -> str:
    """Start a background contract review. Returns a job ID immediately."""
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "running", "result": None}

    # Launch subagent in a background thread
    thread = threading.Thread(
        target=run_review_in_background,
        args=(job_id, document)
    )
    thread.daemon = True
    thread.start()

    return f"Review started. Job ID: {job_id}. I'll keep this running in the background."


@tool
def check_status(job_id: str) -> str:
    """Check the status of a background review job."""
    job = jobs.get(job_id)
    if not job:
        return f"No job found with ID: {job_id}"
    return job["status"]  # "running" | "completed" | "failed"


@tool
def get_result(job_id: str) -> str:
    """Retrieve the completed result of a review job."""
    job = jobs.get(job_id)
    if not job:
        return f"No job found with ID: {job_id}"
    if job["status"] != "completed":
        return f"Job {job_id} is not finished yet (status: {job['status']})"
    return job["result"]


# ------------------------------------------------------------------
# Main agent with all 3 async tools
# ------------------------------------------------------------------

main_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[start_review, check_status, get_result],
    system_prompt="You manage long-running document review tasks. "
                  "Start jobs in the background, check status when asked, "
                  "and retrieve results when complete."
)

# ------------------------------------------------------------------
# Conversation flow
# ------------------------------------------------------------------

# Turn 1: user submits the contract
response1 = main_agent.invoke({
    "messages": [{"role": "user", "content": "Please review this M&A contract: [document text...]"}]
})
print(response1["messages"][-1].content)
# → "Review started. Job ID: a3f9c1b2. This may take a few minutes."

# Turn 2: user asks for status (can happen any time)
response2 = main_agent.invoke({
    "messages": [
        {"role": "user",    "content": "Review this contract: [doc]"},
        {"role": "assistant","content": "Review started. Job ID: a3f9c1b2."},
        {"role": "user",    "content": "Is it done yet?"},
    ]
})
print(response2["messages"][-1].content)
# → "Still running. The review is in progress."

# Turn 3: user checks again after job finishes
response3 = main_agent.invoke({
    "messages": [
        # ... full history ...
        {"role": "user", "content": "Check job a3f9c1b2 and summarise the results."},
    ]
})
print(response3["messages"][-1].content)
# → "Review complete. Key findings: [detailed analysis]"