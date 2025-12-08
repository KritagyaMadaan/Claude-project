import os
import random
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gtts import gTTS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# CORRECT CORS CONFIGURATION (Let the library do the work)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================
# CONFIGURATION
# ============================
OPENROUTER_API_KEY = "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"

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

USERS = ["Kritagya", "Atul", "Piyush", "Shivam"]
ISSUES = ["Brake Pad Wear", "Engine Overheat", "Low Oil Pressure", "Battery Critical", "Tire Puncture"]

@app.route("/", methods=["GET"])
def home():
    return "✅ Backend Online"

@app.route("/api/start-journey", methods=["POST"])
def start_journey():
    print("--> 🚀 Request Received!")

    try:
        selected_user = random.choice(USERS)
        selected_issue = random.choice(ISSUES)
        
        # AI SCRIPT
        print(f"--> Generating Script for {selected_user}...")
        try:
            prompt = ChatPromptTemplate.from_template(
                "You are a vehicle AI. User: {owner}. Issue: {issue}. "
                "Write a very short (1 sentence) urgent alert."
            )
            chain = prompt | llm
            ai_response = chain.invoke({"owner": selected_user, "issue": selected_issue})
            voice_script = ai_response.content
        except Exception as e:
            print(f"⚠️ AI Error (using fallback): {e}")
            voice_script = f"Alert for {selected_user}. {selected_issue} detected."

        # AUDIO
        print("--> Generating Audio...")
        audio_dir = os.path.join(os.path.dirname(__file__), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        filename = f"voice_{uuid.uuid4()}.mp3"
        filepath = os.path.join(audio_dir, filename)
        tts = gTTS(text=voice_script, lang="en")
        tts.save(filepath)

        # RESPONSE
        print("--> Sending Response...")
        return jsonify({
            "success": True,
            "data": {
                "owner": selected_user,
                "vehicle": "Vehicle-01",
                "issue": selected_issue,
                "risk": "critical",
                "audio_url": f"{request.host_url}audio/{filename}"
            }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), "audio"), filename)

if __name__ == "__main__":
    # Using Port 5000 is fine now that headers are fixed
    print("🚀 Server starting on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
