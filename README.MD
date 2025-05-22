# Stateless Chat Graph (LangGraph + OpenAI)

A minimal LangGraph project that uses OpenAI to build a stateless chat agent with one tool: `get_current_time`.

---

## ⚠️ Setup Note

Before running, make sure to update the following variables in `main.py`:

```python
OPENAI_API_KEY = "your-api-key"     # Replace with your actual OpenAI key
PROXY = "http://your-proxy:port"    # Replace with your proxy or set to None if not needed
``` 

## 📦 Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
langgraph dev
```
