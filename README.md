# Conversation Pipeline for Therapod

Pipeline that handles the conversation history, RAG, and response of Therapod 

## Installation

First install Ollama to pull a model for demo (visit their website for the installer)

Then run this on the Ollama CLI
```bash
  ollama run yi:6b-chat-v1.5-q4_K_M
```

Then install requirements (create environment first)
```bash
  # python -m venv myenv (for environment)
  pip install requirements.txt
```

Run the program 
```bash
  python conversation-main.py
```
