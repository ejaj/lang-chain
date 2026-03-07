# TYPE: Initialize Model
# DESCRIPTION: Two ways to start a model.
# init_chat_model → easy shortcut, works with any provider
# Model class (ChatOpenAI etc) → full control over settings

import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key"

# --- Option A: init_chat_model (recommended, works with any provider) ---
model = init_chat_model(
    "gpt-4.1",
    model_provider="openai",
    temperature=0.7,    # 0=focused, 1=creative
    max_tokens=1000,    # max length of reply
    timeout=30,         # stop waiting after 30 seconds
    max_retries=6,      # retry if network fails
)

# --- Option B: direct class (OpenAI specific) ---
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
)

# --- Both work exactly the same way after this ---
response = model.invoke("Why do parrots talk?")
print(response.content)
# → "Parrots talk because..."