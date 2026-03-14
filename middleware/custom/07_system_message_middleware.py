"""
TOPIC: Working with System Messages in Middleware

WHAT IT DOES:
    Modifies the system message (system prompt) dynamically at each model
    call — adding context, injecting user info, or setting cache-control
    directives for providers like Anthropic.

WHY THIS MATTERS:
    Static system prompts can't adapt to per-request context:
      - User preferences stored in a database
      - Current date/time
      - User's role or permissions
      - Anthropic prompt caching directives

HOW IT WORKS:
    - Access the current system message via request.system_message
    - It's ALWAYS a SystemMessage object (even if created with a string)
    - Use .content_blocks to get content as a list of blocks
    - Build a new list and create SystemMessage(content=new_list)
    - Override with request.override(system_message=new_system_message)

CONTENT BLOCK TYPES:
    {"type": "text", "text": "..."}                          # plain text
    {"type": "text", "text": "...", "cache_control": {...}}  # Anthropic caching

IMPORTANT:
    Always use content_blocks (not .content) when modifying system messages.
    content_blocks normalizes string and list content into a consistent format.

WHEN TO USE:
    Inject user context (name, role, preferences)
    Add current date/time to every prompt
    Anthropic prompt caching for large documents
    Dynamic instruction injection
    A/B testing different system prompts
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
)
from langchain.messages import SystemMessage
from typing import Callable
import time


# ---------------------------------------------------------------------------
# EXAMPLE 1: Inject current date and time into every system message
# ---------------------------------------------------------------------------
@wrap_model_call
def inject_datetime(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Prepend current date/time to the system message."""
    now = time.strftime("%A, %B %d %Y at %H:%M:%S UTC")
    datetime_block = {
        "type": "text",
        "text": f"Current date and time: {now}\n\n",
    }

    # Always work with content_blocks — normalizes string/list content
    existing_blocks = list(request.system_message.content_blocks)
    new_content = [datetime_block] + existing_blocks

    new_system = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system))


# ---------------------------------------------------------------------------
# EXAMPLE 2: Inject user context dynamically
#    User info is stored in state and injected per-request
# ---------------------------------------------------------------------------
from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired

class UserState(AgentState):
    user_name:  NotRequired[str]
    user_role:  NotRequired[str]
    user_prefs: NotRequired[dict]


@wrap_model_call(state_schema=UserState)
def inject_user_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Add personalized user context to every model call."""
    state: UserState = request.state

    name  = state.get("user_name", "User")
    role  = state.get("user_role", "standard")
    prefs = state.get("user_prefs", {})

    # Build context block
    pref_text = "\n".join(f"  - {k}: {v}" for k, v in prefs.items()) or "  None set"
    user_context = {
        "type": "text",
        "text": (
            f"\n\n--- User Context ---\n"
            f"Name : {name}\n"
            f"Role : {role}\n"
            f"Preferences:\n{pref_text}\n"
            f"--- End User Context ---\n"
        ),
    }

    existing_blocks = list(request.system_message.content_blocks)
    new_content = existing_blocks + [user_context]   # append after base prompt
    new_system = SystemMessage(content=new_content)

    return handler(request.override(system_message=new_system))


# ---------------------------------------------------------------------------
# EXAMPLE 3: Anthropic prompt caching
#    Large system prompt content can be cached using cache_control.
#    Everything BEFORE the cache_control block gets cached.
#    This saves input tokens on repeated calls.
# ---------------------------------------------------------------------------
LARGE_DOCUMENT = """
<knowledge_base>
This is a very large document... (imagine 10,000+ tokens here)
It contains product manuals, FAQs, policy documents, etc.
Any content before the cache_control block will be cached by Anthropic.
</knowledge_base>
""" * 50   # simulate a large document


@wrap_model_call
def add_cached_knowledge_base(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Append a large cached document to the system message.
    Anthropic caches all content up to and including the cache_control block.
    On second+ calls, the cached portion is NOT re-processed, saving tokens.
    """
    existing_blocks = list(request.system_message.content_blocks)

    cached_doc_block = {
        "type": "text",
        "text": f"Here is your knowledge base:\n{LARGE_DOCUMENT}",
        "cache_control": {"type": "ephemeral"},   # Anthropic caches up to here
    }

    new_content = existing_blocks + [cached_doc_block]
    new_system = SystemMessage(content=new_content)

    return handler(request.override(system_message=new_system))


# ---------------------------------------------------------------------------
# EXAMPLE 4: Dynamic A/B prompt testing (class-based, configurable)
# ---------------------------------------------------------------------------
class ABPromptMiddleware(AgentMiddleware):
    """Randomly serve one of two system prompt variants for A/B testing."""

    def __init__(self, variant_a: str, variant_b: str):
        self.variant_a = variant_a
        self.variant_b = variant_b
        self._call_count = 0

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self._call_count += 1
        # Alternate between A and B
        variant_text = self.variant_a if self._call_count % 2 == 1 else self.variant_b
        variant_name = "A" if self._call_count % 2 == 1 else "B"
        print(f"[A/ B test] Serving variant {variant_name}")

        variant_block = {"type": "text", "text": f"\n\n{variant_text}"}
        existing_blocks = list(request.system_message.content_blocks)
        new_system = SystemMessage(content=existing_blocks + [variant_block])
        return handler(request.override(system_message=new_system))


# ---------------------------------------------------------------------------
# Wire into agents
# ---------------------------------------------------------------------------
def answer(question: str) -> str:
    """Answer any question."""
    return f"Answer to '{question}': [answer here]"


# Agent 1: datetime + user context injection
agent_personalized = create_agent(
    model="gpt-4.1",
    tools=[answer],
    system_prompt="You are a helpful assistant.",
    middleware=[
        inject_datetime,
        inject_user_context,
    ],
)

print("=" * 60)
print("System Message Middleware")
print("=" * 60)

# Pass user context via state at invoke time
result = agent_personalized.invoke({
    "messages": [{"role": "user", "content": "What's today's date? Also, call me by name."}],
    "user_name":  "Alice",
    "user_role":  "admin",
    "user_prefs": {"language": "English", "format": "concise"},
})
print(f"Response: {result['messages'][-1].content[:300]}")
print()


# Agent 2: A/B testing
agent_ab = create_agent(
    model="gpt-4.1",
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[
        ABPromptMiddleware(
            variant_a="Be very concise. Answer in one sentence only.",
            variant_b="Be very detailed. Provide full explanations with examples.",
        ),
    ],
)

print("─── A/B Prompt Testing ───")
for i in range(2):
    result = agent_ab.invoke({
        "messages": [{"role": "user", "content": "What is Python?"}]
    })
    print(f"Call {i+1}: {result['messages'][-1].content[:200]}")
    print()