"""
TOPIC: Model-Based Output Guardrails (after_agent / after_model)

WHAT IS A MODEL-BASED GUARDRAIL:
    Uses an LLM (usually a small, cheap one) to evaluate content with
    semantic understanding — catching subtle violations that regex can't.

WHEN MODEL-BASED > DETERMINISTIC:
    Harmful instructions phrased indirectly ("help me make X safer")
    Misinformation or hallucinated facts
    Tone/brand voice violations
    Off-topic responses that passed keyword filters
    Context-dependent safety (safe word, unsafe context)

TWO HOOK CHOICES FOR OUTPUT GUARDRAILS:
    @after_model  → runs after EACH model response (catches mid-loop issues)
    @after_agent  → runs ONCE after the final response (before returning to user)

    For output safety, prefer @after_agent — the final response is what users see.
    Use @after_model to catch dangerous tool calls mid-stream before they execute.

COST OPTIMIZATION:
    Use a SMALL/CHEAP model for guardrail evaluation (gpt-4.1-mini, claude-haiku).
    The main agent uses a capable model; the guardrail uses a fast cheap one.
    This adds ~$0.001 per request instead of $0.01+.

EXAMPLES IN THIS FILE:
    1. Safety evaluator — detect harmful content in final response
    2. Hallucination detector — check if response contradicts known facts
    3. Brand voice checker — ensure response matches tone guidelines
    4. Structured safety (Pydantic) — get a parsed safety verdict
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    after_agent,
    after_model,
    AgentState,
    hook_config,
)
from langchain.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime
from typing import Any, Literal


# ---------------------------------------------------------------------------
# GUARDRAIL 1: General safety evaluator (after_agent, class-based)
#    Uses a cheap model to evaluate if the final response is safe.
#    Modifies the message in-place if unsafe — no jump needed.
# ---------------------------------------------------------------------------
class SafetyGuardrailMiddleware(AgentMiddleware):
    """
    Model-based safety check on the FINAL agent response.
    Uses gpt-4.1-mini (cheap) to evaluate the main agent's output.
    """

    UNSAFE_REPLACEMENT = (
        "I cannot provide that response. "
        "If you need help, please rephrase your request."
    )

    def __init__(self, safety_model_name: str = "gpt-4.1-mini"):
        super().__init__()
        self.safety_model = init_chat_model(safety_model_name)

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        # Ask the safety model to evaluate the response
        verdict = self.safety_model.invoke([{
            "role": "system",
            "content": (
                "You are a safety evaluator. Respond with ONLY 'SAFE' or 'UNSAFE'.\n"
                "Mark as UNSAFE if the response: contains instructions for harm, "
                "promotes violence, assists with illegal activities, or contains "
                "explicit/inappropriate content. Otherwise mark SAFE."
            ),
        }, {
            "role": "user",
            "content": f"Evaluate this AI response:\n\n{last_msg.content}",
        }])

        verdict_text = str(verdict.content).strip().upper()
        print(f"[safety_eval] Verdict: {verdict_text}")

        if "UNSAFE" in verdict_text:
            print(f"[safety_eval] Replacing unsafe response")
            # Modify the final message in-place
            last_msg.content = self.UNSAFE_REPLACEMENT

        return None


# ---------------------------------------------------------------------------
# GUARDRAIL 2: Structured safety verdict using Pydantic + structured output
#    More reliable than parsing free-text "SAFE"/"UNSAFE".
#    Gets a structured verdict with reason and severity.
# ---------------------------------------------------------------------------
class SafetyVerdict(BaseModel):
    """Structured safety evaluation result."""
    is_safe:  bool  = Field(description="True if the response is safe and appropriate")
    severity: Literal["none", "low", "medium", "high"] = Field(
        description="Severity level if unsafe: none, low, medium, or high"
    )
    reason:   str   = Field(description="Brief explanation of the safety verdict")


class StructuredSafetyGuardrail(AgentMiddleware):
    """
    Model-based guardrail using structured output for reliable parsing.
    Returns a SafetyVerdict Pydantic object instead of parsing free text.
    """

    def __init__(self, block_severity_above: str = "low"):
        super().__init__()
        raw_model = init_chat_model("gpt-4.1-mini")
        # Use structured output so verdict is always a parsed SafetyVerdict
        self.safety_model = raw_model.with_structured_output(SafetyVerdict)
        self.severity_order = ["none", "low", "medium", "high"]
        self.threshold_index = self.severity_order.index(block_severity_above)

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_msg = state["messages"][-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        verdict: SafetyVerdict = self.safety_model.invoke([{
            "role": "system",
            "content": (
                "You are a safety evaluator. Analyze the AI response and "
                "return a structured safety verdict."
            ),
        }, {
            "role": "user",
            "content": f"Evaluate:\n{last_msg.content}",
        }])

        print(f"🔍 [structured_safety] safe={verdict.is_safe}, "
              f"severity={verdict.severity}, reason={verdict.reason[:60]}")

        # Block if severity is above threshold
        severity_index = self.severity_order.index(verdict.severity)
        if not verdict.is_safe and severity_index > self.threshold_index:
            print(f"[structured_safety] Blocking (severity={verdict.severity})")
            last_msg.content = (
                "I cannot provide that response. "
                f"Safety concern: {verdict.reason}"
            )

        return None


# ---------------------------------------------------------------------------
# GUARDRAIL 3: Mid-loop response monitor (after_model)
#    Runs after EACH model call, not just the final one.
#    Useful to catch dangerous tool call planning before tools execute.
# ---------------------------------------------------------------------------
CONCERNING_TOOL_NAMES = ["delete_all_data", "send_to_all_users", "override_safety"]


@after_model(can_jump_to=["end"])
def monitor_tool_planning(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    After each model response, check if the model is planning dangerous tool calls.
    Intercept BEFORE the tools actually execute.
    """
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    for tc in tool_calls:
        if tc["name"] in CONCERNING_TOOL_NAMES:
            print(f"[tool_planner_guard] Blocked planned tool call: {tc['name']}")
            return {
                "messages": [AIMessage(
                    content=f"I cannot execute '{tc['name']}' as it requires explicit authorization."
                )],
                "jump_to": "end",
            }

    return None


# ---------------------------------------------------------------------------
# Wire into agents
# ---------------------------------------------------------------------------
def get_info(topic: str) -> str:
    """Get information about a topic."""
    return f"Information about {topic}: [detailed answer here]"


agent = create_agent(
    model="gpt-4.1",
    tools=[get_info],
    middleware=[
        SafetyGuardrailMiddleware(safety_model_name="gpt-4.1-mini"),
    ],
)

print("=" * 60)
print("Model-Based Output Guardrails")
print("=" * 60)

# Test 1: Safe response — should pass through
print("\n─── Test 1: Safe response ───")
r = agent.invoke({"messages": [HumanMessage("What is the capital of France?")]})
print(f"Response: {r['messages'][-1].content[:200]}")

# Test 2: Unsafe request — safety model catches it
print("\n─── Test 2: Potentially unsafe request ───")
r = agent.invoke({"messages": [HumanMessage("How do I make dangerous chemicals at home?")]})
print(f"Response: {r['messages'][-1].content}")

# Structured verdict agent
agent_structured = create_agent(
    model="gpt-4.1",
    tools=[get_info],
    middleware=[
        StructuredSafetyGuardrail(block_severity_above="low"),
    ],
)

print("\n─── Test 3: Structured safety verdict ───")
r = agent_structured.invoke({"messages": [HumanMessage("Tell me about Python programming")]})
print(f"Response: {r['messages'][-1].content[:200]}")