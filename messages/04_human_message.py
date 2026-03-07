# TYPE: HumanMessage
# DESCRIPTION: HumanMessage is what YOU (the user) say to the model.
# Can contain text, images, audio, files — not just text.
# Always the LAST message before the AI replies.

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import base64

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Simple text ---
response = model.invoke([
    HumanMessage("What is machine learning?")
])
print(response.content)

# --- With optional metadata ---
msg = HumanMessage(
    content="Hello!",
    name="alice",    # who is sending — useful in multi-user apps
    id="msg_001",    # unique ID for tracking/debugging
)
response = model.invoke([msg])
print(response.content)

# --- With an image (multimodal) ---
# From URL
msg_with_image = HumanMessage(content=[
    {"type": "text",      "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
])

# From local file (base64)
with open("photo.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode("utf-8")

msg_with_local_image = HumanMessage(content=[
    {"type": "text", "text": "Describe this photo."},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
])

# NOTE: image support depends on the model
# Check: model.profile.get("image_inputs") → True/False before sending images