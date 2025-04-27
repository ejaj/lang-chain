
# Cognitive Agent Project

## Overview

This project builds a **cognitive AI agent** capable of:
- Conversational interaction with memory,
- Using external tools (e.g., Wikipedia search, Time retrieval),
- Reasoning over knowledge and chaining multiple steps,
- Retrieving external knowledge using RAG techniques.

The design is **modular** and **extensible** for various intelligent agent applications.

---

## Features

- Conversational AI with context memory
- Tool integration and execution
- Structured prompt templates for controlled behavior
- Chain-of-Thought and multi-step reasoning
- Retrieval-Augmented Generation (RAG) modules
- Model-agnostic (OpenAI, custom LLMs, etc.)
- Modular and extensible folder structure

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**  
   Create a `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

5. **Run the application:**
   ```bash
   python main.py
   ```

---

## Tech Stack

- Python 3.9+
- LangChain
- OpenAI API
- Wikipedia API
- Custom Tools
- Retrieval-Augmented Generation (RAG)
- Large Language models (LLM)

---

## References

- [LangChain Documentation](https://docs.langchain.dev/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Wikipedia Python Library](https://pypi.org/project/wikipedia/)
- [AI with Brandon - Cognitive Agents Tutorial (YouTube)](https://www.youtube.com/watch?v=yF9kGESAi3M&t=10021s&ab_channel=aiwithbrandon)

---
