# TYPE: Structured Output
# DESCRIPTION: Force the model to reply in a specific shape (like a form).
# Instead of free text, you get back a real Python object with named fields.
# Use when you need to extract or generate data in a predictable format.

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import TypedDict

model = init_chat_model("gpt-4.1", model_provider="openai")

# --- Option A: Pydantic (recommended — has validation + descriptions) ---
class Movie(BaseModel):
    title: str     = Field(..., description="Title of the movie")
    year: int      = Field(..., description="Year it was released")
    director: str  = Field(..., description="Director's name")
    rating: float  = Field(..., description="Rating out of 10")


model_structured = model.with_structured_output(Movie)

response = model_structured.invoke("Tell me about the movie Inception")
print(response.title)    # → "Inception"
print(response.year)     # → 2010
print(response.director) # → "Christopher Nolan"
print(response.rating)   # → 8.8
# response is a real Movie object — not just text!

# --- Option B: TypedDict (simpler, no validation) ---
class Person(TypedDict):
    name: str
    age: int
    job: str

model_structured2 = model.with_structured_output(Person)
response2 = model_structured2.invoke("Describe a software engineer named Alice, age 30")
print(response2["name"])  # → "Alice"
print(response2["age"])   # → 30
print(response2["job"])   # → "Software Engineer"

# KEY POINT: without structured output → free text you have to parse yourself
#            with structured output    → clean Python object, ready to use