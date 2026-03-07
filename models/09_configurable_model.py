# TYPE: Configurable Model
# DESCRIPTION: Create one model object that can switch providers at runtime.
# Instead of creating separate model objects for each provider,
# you create one and pass which model to use when invoking.
# Great for A/B testing or letting users pick their preferred model.

from langchain.chat_models import init_chat_model

# --- Create one model with no provider locked in ---
flexible_model = init_chat_model(temperature=0)  # no model specified!

# --- Use different models at runtime ---
response_1 = flexible_model.invoke(
    "What's your name?",
    config={"configurable": {"model": "gpt-4.1-mini"}}   # ← use OpenAI
)
print(response_1.content)

response_2 = flexible_model.invoke(
    "What's your name?",
    config={"configurable": {"model": "claude-sonnet-4-6"}}  # ← use Anthropic
)
print(response_2.content)

response_3 = flexible_model.invoke(
    "What's your name?",
    config={"configurable": {"model": "gemini-2.0-flash", "model_provider": "google"}}
)
print(response_3.content)

# --- Compare vs the old way (creating separate objects) ---
# OLD WAY — messy, lots of objects
# model_openai    = ChatOpenAI(model="gpt-4.1-mini")
# model_anthropic = ChatAnthropic(model="claude-sonnet-4-6")
# model_google    = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# NEW WAY — one object, switch at runtime
# flexible_model with config={"configurable": {"model": "..."}}