# TYPE: Dynamic Model
# Automatically swap to a smarter model when conversation gets long.
# Saves cost on simple chats, upgrades when needed.

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.messages import SystemMessage, HumanMessage, AIMessage

cheap_model = ChatOpenAI(model="gpt-4.1-mini")  # fast & cheap
smart_model = ChatOpenAI(model="gpt-4.1")        # slower & smarter

@wrap_model_call
def pick_model(request, handler):
    msg_count = len(request.state["messages"])

    if msg_count > 10:
        # Long conversation → upgrade
        return handler(request.override(model=smart_model))
    else:
        # Short conversation → stay cheap
        return handler(request.override(model=cheap_model))

agent = create_agent(
    model=cheap_model,
    tools=[],
    middleware=[pick_model],
)

# Messages 1-10  → uses gpt-4.1-mini (cheap)
# Messages 11+   → uses gpt-4.1 (smart), automatically

# result = agent.invoke({
#     "messages": [{"role": "user", "content": "What is 42 * 17?"}]
# })
result = agent.invoke({
    "messages": [
        SystemMessage("You are a math tutor."),  #  rules - ALWAYS FIRST
        HumanMessage("What is 2+2?"),            # you talk
        AIMessage("The answer is 4."),           # AI replied (history)
        HumanMessage("And 3+3?"),                # you talk again ← ALWAYS LAST
    ]                                            #  AI will reply to this
})

print(result["messages"])