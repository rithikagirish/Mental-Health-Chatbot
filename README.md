# Mental Health Support Chatbot (Groq-powered)

A lightweight Flask backend and static frontend for a mental-health support chatbot that uses a Groq-compatible LLM. The backend classifies user emotion, generates empathetic responses, and produces mood insights for the frontend.

Files
- [backend_groqwinsights.py](backend_groqwinsights.py) — main Flask app and core logic, includes [`GroqClient`](backend_groqwinsights.py), [`ConversationContext`](backend_groqwinsights.py), and [`MentalHealthChatbot`](backend_groqwinsights.py).
- [mental_health_frontend.html](mental_health_frontend.html) — single-file frontend UI and client-side JS.

Requirements
- Python 3.8+
- pip packages: Flask, flask-cors, python-dotenv, requests

Setup

```bash
# Install dependencies
pip install flask flask-cors python-dotenv requests

```

Environment

The backend expects a Groq API key in the environment variable GROQ_API_KEY. You can provide it via a .env file or your shell environment.

Run (development)

Start the backend:
```bash
python [backend_groqwinsights.py](http://_vscodecontentref_/0)
//Open the frontend:
//Visit http://localhost:5000 in your browser (the backend serves mental_health_frontend.html at /), or open mental_health_frontend.html directly.
```

GroqClient wraps the external Groq API usage.
MentalHealthChatbot maintains per-session ConversationContext, classifies emotion via LLM, generates responses, and produces the structured mood insights consumed by the UI.
The frontend JS calls /chat, displays messages, and updates the insights panel from the mood_insights object returned by the backend.
Customization pointers

To change the LLM model or parameters: edit GROQ_MODEL and temp settings in backend_groqwinsights.py.
To change emotion categories, adjust MentalHealthChatbot.emotion_categories in backend_groqwinsights.py.
Frontend markup and UX are in mental_health_frontend.html.

Troubleshooting

If requests fail, confirm GROQ_API_KEY is set and reachable from the running environment.
Check console output from backend_groqwinsights.py for exceptions or request errors.
If CORS or connection issues appear when loading the static HTML file from filesystem, prefer opening http://localhost:5000 (backend serves the frontend).
