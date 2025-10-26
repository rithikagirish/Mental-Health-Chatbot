#!/usr/bin/env python3
"""
Flask Backend API for Mental Health Chatbot with Groq Integration
Generates unique responses using Groq LLM API
Detects emotions even without explicit keywords
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
from collections import deque, defaultdict
import time
import os
import requests
import json # Added for better debugging/logging

# import os
# print(os.getenv("GROQ_API_KEY")) # Commented out for cleaner output

from dotenv import load_dotenv
load_dotenv()



# ======== CONFIG ========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # store your key as env var
# Adjusted model name based on common Groq-supported models
# NOTE: The original model 'meta-llama/llama-4-maverick-17b-128e-instruct' is likely a placeholder/non-existent public model.
# Using a common accessible model like llama3-8b-8192 for the example.
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"  

# ======== FLASK APP ========
app = Flask(__name__)
CORS(app)

chatbot_instances = {}


# ========= HELPERS =========
class GroqClient:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key):
        self.api_key = api_key

    def chat(self, messages, temperature=0.7):
        """Call Groq chat completion API"""
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        
        # print(f"Sending request to Groq with model: {GROQ_MODEL}") # Debug line
        
        response = requests.post(self.BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


groq_client = GroqClient(GROQ_API_KEY)


class ConversationContext:
    def __init__(self, max_context=5):
        self.memory = deque(maxlen=max_context)
        self.themes = defaultdict(int)
        # emotions will store the history of classified emotions
        self.emotions = [] 

    def add_turn(self, user, bot, emotion):
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "bot": bot,
            "emotion": emotion
        })
        # Store all classified emotions for balance calculation
        if emotion:
            self.emotions.append(emotion)

    def get_context(self):
        # Format the last 2 turns for context generation
        return [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in self.memory][-2:]


class MentalHealthChatbot:
    def __init__(self):
        self.context = ConversationContext()
        self.session_start = datetime.now()
        self.message_count = 0
        self.emotion_categories = ["depression", "anxiety", "anger", "self_harm", "positive", "neutral"]

    def classify_emotion(self, user_input):
        """Ask Groq LLM to classify emotion"""
        messages = [
            {"role": "system", "content": f"You are an assistant that classifies the primary emotional tone of the user's message. Choose only one word from this list: {', '.join(self.emotion_categories)}."},
            {"role": "user", "content": f"Classify the emotion in this message: {user_input}"}
        ]
        try:
            # Use a low temperature for consistent classification
            result = groq_client.chat(messages, temperature=0.1).strip().lower() 
            
            # Simple check for the classified category
            for category in self.emotion_categories:
                if category in result:
                    return category
            
            return "neutral" # Default fallback
            
        except Exception as e:
            print(f"Error classifying emotion: {e}")
            return "neutral"

    def generate_response(self, user_input, emotion, context):
        """Use Groq to generate unique chatbot response"""
        messages = [
            {"role": "system", "content": "You are a compassionate mental health support chatbot. Be empathetic, supportive, and safe. Keep responses under 120 words."},
            {"role": "system", "content": f"The detected emotion is: {emotion}."},
        ]
        if context:
            messages.append({"role": "system", "content": f"Here is recent chat context:\n{context}"})
            
        messages.append({"role": "user", "content": user_input})

        try:
            return groq_client.chat(messages, temperature=0.8)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again later."
    
    # NEW METHOD to generate the mood insights structure expected by the frontend
    def _generate_mood_insights(self, current_emotion: str):
        insights = {
            "current_mood": current_emotion,
            "risk_level": "low",
            "emotional_balance": {"positive": 0, "neutral": 0, "negative": 0},
            "suggestions": []
        }
        
        # 1. Emotional Balance from history
        positive_emotions = ["positive"]
        negative_emotions = ["depression", "anxiety", "anger", "self_harm"]
        
        recent_emotions = self.context.emotions[-5:] # Last 5 emotions for balance
        
        for emotion in recent_emotions:
            if emotion in positive_emotions:
                insights["emotional_balance"]["positive"] += 1
            elif emotion in negative_emotions:
                insights["emotional_balance"]["negative"] += 1
            else:
                insights["emotional_balance"]["neutral"] += 1
        
        # 2. Risk Level and Suggestions based on current emotion
        if current_emotion == "self_harm":
            insights["risk_level"] = "high"
            insights["suggestions"] = [
                "🚨 Please reach out to a crisis hotline immediately.",
                "Your life has value, and there is help available.",
                "Consider speaking with a mental health professional."
            ]
            # Flag for urgent intervention, though the frontend code doesn't explicitly use it via this key, it's good practice.
            insights["urgent_intervention_needed"] = True 
            
        elif current_emotion in ["depression", "anxiety", "anger"]:
            insights["risk_level"] = "moderate"
            insights["suggestions"] = [
                f"It's okay to feel {current_emotion}. Try to take a few deep breaths.",
                "Consider taking some time for self-care today.",
                "Reach out to someone you trust to talk."
            ]
            
        elif current_emotion == "positive":
            insights["suggestions"] = [
                "That's wonderful! What can you do to keep this positive feeling going?",
                "Remember to savor these moments of joy.",
                "Keep practicing self-awareness and gratitude."
            ]
            
        else: # Neutral
            insights["suggestions"] = [
                "How you're feeling is valid and important.",
                "I'm here to listen to whatever you'd like to share.",
                "Sometimes just having someone to talk to can be helpful."
            ]
        
        return insights


    def process_message(self, user_input):
        start = time.time()

        # Detect emotion
        emotion = self.classify_emotion(user_input)

        # Get context
        context_text = "\n".join(self.context.get_context()) # Use get_context() which returns the last 2 turns

        # Generate response
        bot_response = self.generate_response(user_input, emotion, context_text)

        # Update memory
        self.context.add_turn(user_input, bot_response, emotion)

        # Generate mood insights for the frontend
        mood_insights = self._generate_mood_insights(emotion)
        
        # Check for urgent intervention from generated insights
        urgent_intervention = mood_insights.get("urgent_intervention_needed", False)

        self.message_count += 1
        duration = str(datetime.now() - self.session_start).split('.')[0]

        return {
            "success": True,
            "data": {
                "bot_response": bot_response,
                # Frontend expects mood_insights, not just emotion_classification
                "mood_insights": mood_insights, 
                # Also include the intervention flag for frontend logic
                "urgent_intervention_needed": urgent_intervention, 
                "session_stats": {
                    "message_count": self.message_count,
                    "session_duration": duration,
                }
            }
        }


# ========= ROUTES =========
@app.route('/')
def index():
    # Renders the HTML frontend file
    try:
        with open('mental_health_frontend.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "Frontend HTML file not found.", 404


@app.route('/chat', methods=['POST'])
def chat():
    # print("Received a POST request to /chat") # Debug line
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        # print("Invalid request") # Debug line
        return jsonify({"success": False, "error": "Invalid request"}), 400

    user_message = data["message"].strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"success": False, "error": "Empty message"}), 400
    
    # print(f"Processing message for session {session_id}: {user_message}") # Debug line

    if not GROQ_API_KEY:
         return jsonify({"success": False, "error": "GROQ_API_KEY not configured"}), 500

    if session_id not in chatbot_instances:
        chatbot_instances[session_id] = MentalHealthChatbot()

    chatbot = chatbot_instances[session_id]
    
    try:
        response = chatbot.process_message(user_message)
        # print(f"Sending response: {json.dumps(response, indent=2)}") # Debug line
        return jsonify(response)
    except Exception as e:
        print(f"Chat processing error: {e}")
        # Return an error structure that still satisfies the frontend's expectations for 'mood_insights'
        return jsonify({
            "success": False,
            "error": "Server error during chat processing.",
            "data": {
                "bot_response": "I'm having a technical issue. Please try again.",
                "mood_insights": {
                    "current_mood": "neutral",
                    "risk_level": "low",
                    "emotional_balance": {"positive": 0, "neutral": 1, "negative": 0},
                    "suggestions": ["Please try again in a moment"]
                },
                "urgent_intervention_needed": False
            }
        }), 500


@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route('/reset_session', methods=['POST'])
def reset_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    chatbot_instances.pop(session_id, None)
    return jsonify({"success": True, "message": "Session reset successfully"})


if __name__ == '__main__':
    print("🚀 Starting Groq-powered Mental Health Chatbot Backend")
    if not GROQ_API_KEY:
        print("⚠️ WARNING: GROQ_API_KEY is not set. Chat functionality will fail.")
    print("Backend available at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)