# ============================
# IMPORTS & CONFIG
# ============================

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import json
import uuid

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from gtts import gTTS

# ============================================
# OPENROUTER / LLM CONFIG
# ============================================
from dotenv import load_dotenv  # <-- ADD THIS

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2-speciale",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ============================
# AGENT STATE DEFINITIONS
# ============================

class JourneyType(str, Enum):
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    EDGE_CASE = "edge_case"


class AgentState(str, Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    RISK_HIGH = "risk_high"
    CALLING = "calling"
    LISTENING = "listening"
    OFFERING_SLOTS = "offering_slots"
    BOOKING = "booking"
    BOOKED = "booked"
    DECLINED = "declined"
    NO_SHOW = "no_show"
    REMINDER = "reminder"
    COMPLETED = "completed"


@dataclass
class Vehicle:
    vehicle_id: str
    make: str
    model: str
    year: int
    owner_name: str
    owner_phone: str
    owner_email: str
    mileage: int = 0
    status: str = "healthy"
    risk_score: float = 0.0
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    last_telemetry: Optional[datetime] = None


@dataclass
class Telemetry:
    engine_temp: float
    battery_health: float
    brake_wear: float
    oil_pressure: float
    dtc_codes: List[str] = field(default_factory=list)
    mileage: int = 0


@dataclass
class Appointment:
    appointment_id: str
    vehicle_id: str
    owner_name: str
    owner_phone: str
    date: datetime
    time: str
    service_center: str
    reason: str
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationState:
    journey_id: str
    vehicle_id: str
    owner_phone: str
    owner_name: str
    predicted_issue: str
    risk_level: str
    state: AgentState
    telemetry_data: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

    call_sid: Optional[str] = None
    available_slots: List[str] = field(default_factory=list)
    selected_slot: Optional[str] = None
    conversation_turns: int = 0
    voice_script: str = ""
    response: str = ""
    booking_confirmed: bool = False
    declined: bool = False
    error: Optional[str] = None

    security_flags: List[str] = field(default_factory=list)


# ============================
# DUMMY XGB MODEL
# ============================

xgb_model_path = "xgb_failure_model.pkl"
if not os.path.exists(xgb_model_path):
    X = np.random.rand(200, 6)
    y = np.random.randint(0, 2, 200)
    model = XGBClassifier()
    model.fit(X, y)
    joblib.dump(model, xgb_model_path)

xgb_model = joblib.load(xgb_model_path)


def preprocess_telemetry(telemetry):
    df = pd.DataFrame([telemetry])
    df = df.fillna(0)
    return df


def predict_failure_node(state: ConversationState) -> ConversationState:
    telemetry = state.telemetry_data
    df = preprocess_telemetry(telemetry)
    prob = float(xgb_model.predict_proba(df)[0][1])

    if prob > 0.85:
        level = "critical"
    elif prob > 0.60:
        level = "high"
    elif prob > 0.40:
        level = "medium"
    else:
        level = "low"

    state.predicted_issue = telemetry.get("issue", "General component wear")
    state.risk_level = level
    state.risk_score = prob
    return state


# ============================
# RCA / CAPA NODE
# ============================

def rca_capa_node(state: ConversationState) -> ConversationState:
    prompt = ChatPromptTemplate.from_template(
        """
Based on telemetry and predicted issue:
Telemetry: {telemetry}
Issue: {issue}
Risk: {risk}

Generate JSON with:
- root_cause
- corrective_action
- preventive_action
"""
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "telemetry": state.telemetry_data,
            "issue": state.predicted_issue,
            "risk": state.risk_level,
        }
    )
    state.rca_capa = getattr(response, "content", "")
    return state


# ============================
# UEBA SECURITY NODE
# ============================

def ueba_security_node(state: ConversationState) -> ConversationState:
    flags: List[str] = []

    if state.risk_score < 0.2 and state.telemetry_data.get("mileage", 0) > 50000:
        flags.append("Telemetry anomaly: mileage inconsistent with risk score")

    if state.conversation_turns > 7:
        flags.append("Suspicious long interaction")

    if not state.owner_phone.startswith("+"):
        flags.append("Invalid phone region")

    state.security_flags = flags
    return state


# ============================
# RISK ANALYSIS & FLOW NODES
# ============================

def analyze_risk_node(state: ConversationState) -> ConversationState:
    if state.risk_level == "critical":
        state.state = AgentState.RISK_HIGH
        prompt = ChatPromptTemplate.from_template(
            """
You are a vehicle maintenance AI agent. A CRITICAL issue has been detected.

Vehicle: {owner_name}'s {vehicle}
Issue: {issue}
Risk Level: {risk_level}

Generate an urgent but professional voice script (2-3 sentences) to alert the owner
that their vehicle has a critical issue requiring emergency service within 24-48 hours.
"""
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            """
You are a vehicle maintenance AI agent offering proactive maintenance.

Vehicle: {owner_name}'s {vehicle}
Issue: {issue}
Risk Level: {risk_level}

Generate a warm, proactive voice script (3-4 sentences) explaining the potential issue
and offering nearby service slots.
"""
        )

    chain = prompt | llm
    response = chain.invoke(
        {
            "owner_name": state.owner_name,
            "vehicle": state.vehicle_id,
            "issue": state.predicted_issue,
            "risk_level": state.risk_level,
        }
    )
    state.voice_script = getattr(response, "content", "")
    return state


def prepare_call_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.CALLING
    print(f"🎙️ Initiating voice call to {state.owner_phone}")
    print(f"📍 Vehicle: {state.vehicle_id}")
    print(f"📢 Script: {state.voice_script}")
    return state


def voice_interaction_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.LISTENING
    state.conversation_turns += 1

    mock_responses = [
        "Yes, I'm interested in booking an appointment.",
        "What dates are available?",
        "I can do Thursday morning.",
        "Thanks, I'll see you then.",
    ]
    if state.conversation_turns <= len(mock_responses):
        owner_response = mock_responses[state.conversation_turns - 1]
    else:
        owner_response = "Thanks, confirmed."

    print(f"👤 Owner: {owner_response}")
    state.response = owner_response
    return state


def route_response(state: ConversationState) -> Literal["declined", "offer_slots", "confirm", "end"]:
    response_lower = state.response.lower()

    if any(word in response_lower for word in ["no", "not interested", "don't want", "cancel", "decline"]):
        return "declined"

    if any(word in response_lower for word in ["yes", "interested", "book", "available", "when", "which time"]):
        return "offer_slots"

    if any(phrase in response_lower for phrase in ["thanks, confirmed", "see you then", "that works", "ok", "okay"]):
        return "confirm"

    return "offer_slots"


def offer_slots_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.OFFERING_SLOTS
    base_date = datetime.now()

    if state.risk_level == "critical":
        slots = [
            (base_date + timedelta(hours=24)).strftime("%A %I:%M %p"),
            (base_date + timedelta(hours=30)).strftime("%A %I:%M %p"),
            (base_date + timedelta(hours=36)).strftime("%A %I:%M %p"),
        ]
    else:
        slots = [
            (base_date + timedelta(days=1)).strftime("%A at 9:00 AM"),
            (base_date + timedelta(days=1)).strftime("%A at 2:00 PM"),
            (base_date + timedelta(days=2)).strftime("%A at 10:00 AM"),
            (base_date + timedelta(days=3)).strftime("%A at 3:00 PM"),
        ]

    state.available_slots = slots
    slots_text = ", ".join(slots)

    prompt = ChatPromptTemplate.from_template(
        """
You are a vehicle maintenance AI agent offering appointment slots.

Available slots for {owner_name}: {slots}

Generate a brief, friendly voice script (2 sentences) offering these slots
and asking which one works best.
"""
    )
    chain = prompt | llm
    response = chain.invoke({"owner_name": state.owner_name, "slots": slots_text})
    state.voice_script = getattr(response, "content", "")
    print(f"🗓️ Offering slots: {slots_text}")
    print(f"📢 Agent: {state.voice_script}")
    return state


def confirm_booking_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.BOOKING
    response_lower = state.response.lower()

    for i, slot in enumerate(state.available_slots):
        if any(day in response_lower for day in ["thursday", "friday", "monday", "tuesday", "wednesday"]):
            state.selected_slot = slot
            break
        if i == 0 and ("first" in response_lower or "morning" in response_lower):
            state.selected_slot = slot
            break
        if i == 1 and ("second" in response_lower or "afternoon" in response_lower):
            state.selected_slot = slot
            break

    if not state.selected_slot and state.available_slots:
        state.selected_slot = state.available_slots[0]

    Appointment(
        appointment_id=f"APT-{datetime.now().timestamp()}",
        vehicle_id=state.vehicle_id,
        owner_name=state.owner_name,
        owner_phone=state.owner_phone,
        date=datetime.now() + timedelta(days=1),
        time=state.selected_slot or "",
        service_center="Downtown Service Center",
        reason=state.predicted_issue,
        status="booked",
    )

    state.booking_confirmed = True
    state.state = AgentState.BOOKED

    prompt = ChatPromptTemplate.from_template(
        """
You are confirming a vehicle service appointment.

Owner: {owner_name}
Vehicle: {vehicle}
Appointment: {date} at {time}
Issue: {issue}
Service Center: Downtown Service Center

Generate a brief confirmation script (2-3 sentences) confirming the appointment,
mentioning the service center, and saying you'll send a reminder text.
"""
    )
    chain = prompt | llm
    date_part = state.selected_slot.split(" at ")[0] if state.selected_slot and " at " in state.selected_slot else state.selected_slot or ""
    time_part = state.selected_slot.split(" at ")[1] if state.selected_slot and " at " in state.selected_slot else ""
    response = chain.invoke(
        {
            "owner_name": state.owner_name,
            "vehicle": state.vehicle_id,
            "date": date_part,
            "time": time_part,
            "issue": state.predicted_issue,
        }
    )

    state.voice_script = getattr(response, "content", "")
    print("✅ Booking Confirmed!")
    print(f"📅 Appointment: {state.selected_slot}")
    print(f"📢 Agent: {state.voice_script}")
    return state


def handle_declined_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.DECLINED
    state.declined = True

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI agent whose appointment offer was declined.

Owner: {owner_name}
Vehicle: {vehicle}
Issue: {issue}

Generate a brief follow-up script (2-3 sentences) acknowledging their decline,
explaining the risk of delaying maintenance, and offering to follow up in 7 days.
"""
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "owner_name": state.owner_name,
            "vehicle": state.vehicle_id,
            "issue": state.predicted_issue,
        }
    )
    state.voice_script = getattr(response, "content", "")
    print("❌ Owner declined appointment")
    print(f"📢 Follow-up: {state.voice_script}")
    print("🔄 Follow-up call scheduled for 7 days")
    return state


def send_reminder_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.REMINDER
    print(f"📱 Sending reminder SMS to {state.owner_phone}")
    print(
        f"📝 Message: Hi {state.owner_name}, reminder: Your {state.vehicle_id} service "
        f"appointment is tomorrow at {state.selected_slot}. Downtown Service Center."
    )
    return state


def handle_no_show_node(state: ConversationState) -> ConversationState:
    state.state = AgentState.NO_SHOW
    state.no_show = True
    print("⚠️ No-show detected for appointment")
    print("📞 Scheduling follow-up call to reschedule")

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI agent following up on a missed appointment.

Owner: {owner_name}
Vehicle: {vehicle}
Missed appointment: {date} at {time}
Issue: {issue}

Generate a brief follow-up script (3 sentences) saying you missed them,
apologizing, and rescheduling as soon as possible, emphasizing urgency.
"""
    )
    chain = prompt | llm
    date_part = ""
    time_part = ""
    if state.selected_slot and " at " in state.selected_slot:
        date_part, time_part = state.selected_slot.split(" at ", 1)

    response = chain.invoke(
        {
            "owner_name": state.owner_name,
            "vehicle": state.vehicle_id,
            "date": date_part or "the appointment",
            "time": time_part,
            "issue": state.predicted_issue,
        }
    )
    state.voice_script = getattr(response, "content", "")
    print(f"📢 Follow-up: {state.voice_script}")
    return state


def handle_multi_vehicle_node(state: ConversationState) -> ConversationState:
    print(f"🚗🚗🚗 Multi-vehicle fleet detected for {state.owner_name}")
    prompt = ChatPromptTemplate.from_template(
        """
You are coordinating service for multiple vehicles.

Owner: {owner_name}
Vehicles: {num_vehicles}
Issues: Multiple critical issues detected

Generate a brief script (3-4 sentences) coordinating batch service for multiple vehicles,
offering combined appointment slots, and explaining cost savings from batch maintenance.
"""
    )
    chain = prompt | llm
    response = chain.invoke(
        {"owner_name": state.owner_name, "num_vehicles": 3}
    )
    state.voice_script = getattr(response, "content", "")
    print(f"📢 Fleet scheduling: {state.voice_script}")
    return state


def handle_data_gaps_node(state: ConversationState) -> ConversationState:
    print(f"⚠️ Data gap detected - incomplete telemetry for {state.vehicle_id}")
    print("🔄 Queueing vehicle for manual telemetry check")
    print("📋 Flagged for call-center review")
    return state


# ============================
# BUILD LANGGRAPH WORKFLOW
# ============================

def create_agent_graph():
    graph = StateGraph(ConversationState)

    graph.add_node("predict_failure", predict_failure_node)
    graph.add_node("rca_capa", rca_capa_node)
    graph.add_node("ueba_security", ueba_security_node)
    graph.add_node("analyze_risk", analyze_risk_node)
    graph.add_node("prepare_call", prepare_call_node)
    graph.add_node("voice_interaction", voice_interaction_node)
    graph.add_node("offer_slots", offer_slots_node)
    graph.add_node("confirm_booking", confirm_booking_node)
    graph.add_node("handle_declined", handle_declined_node)
    graph.add_node("send_reminder", send_reminder_node)
    graph.add_node("handle_no_show", handle_no_show_node)
    graph.add_node("handle_multi_vehicle", handle_multi_vehicle_node)
    graph.add_node("handle_data_gaps", handle_data_gaps_node)

    graph.add_edge(START, "predict_failure")
    graph.add_edge("predict_failure", "rca_capa")
    graph.add_edge("rca_capa", "ueba_security")
    graph.add_edge("ueba_security", "analyze_risk")
    graph.add_edge("analyze_risk", "prepare_call")
    graph.add_edge("prepare_call", "voice_interaction")

    graph.add_conditional_edges(
        "voice_interaction",
        route_response,
        {
            "declined": "handle_declined",
            "offer_slots": "offer_slots",
            "confirm": "confirm_booking",
            "end": END,
        },
    )

    graph.add_edge("offer_slots", "voice_interaction")
    graph.add_edge("confirm_booking", "send_reminder")
    graph.add_edge("handle_declined", END)
    graph.add_edge("send_reminder", END)
    graph.add_edge("handle_no_show", END)
    graph.add_edge("handle_multi_vehicle", END)
    graph.add_edge("handle_data_gaps", END)

    return graph.compile()


def execute_maintenance_journey(journey_data: Dict) -> Dict:
    default_telemetry = {
        "engine_temp": 92.5,
        "battery_health": 65,
        "brake_wear": 0.6,
        "oil_pressure": 28,
        "mileage": 35000,
        "vibration_level": 0.12,
    }

    state = ConversationState(
        journey_id=journey_data["journey_id"],
        vehicle_id=journey_data["vehicle_id"],
        owner_phone=journey_data["owner_phone"],
        owner_name=journey_data["owner_name"],
        predicted_issue=journey_data.get("predicted_issue", ""),
        risk_level=journey_data.get("risk_level", "low"),
        telemetry_data=journey_data.get("telemetry_data", default_telemetry),
        state=AgentState.IDLE,
    )

    agent_graph = create_agent_graph()

    print("\n" + "=" * 60)
    print("🤖 STARTING MAINTENANCE AGENT JOURNEY")
    print(f" Vehicle: {state.vehicle_id} | Risk: {state.risk_level}")
    print("=" * 60 + "\n")

    result = agent_graph.invoke(state)

    print("\n" + "=" * 60)
    print("✅ JOURNEY COMPLETED")
    print(f" Status: {result['state']}")
    print(f" Booked: {result['booking_confirmed']}")
    print(f" Declined: {result['declined']}")
    print("=" * 60 + "\n")

    return result


# ============================
# gTTS AUDIO HELPERS
# ============================

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


def generate_voice_audio(text: str) -> str:
    filename = f"voice_{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    tts = gTTS(text=text, lang="en")
    tts.save(filepath)
    return filename


# ============================
# FLASK APP & ROUTES
# ============================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

active_journeys: Dict[str, Dict] = {}
completed_journeys: Dict[str, Dict] = {}


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,GET,OPTIONS")
    return response


@app.route("/")
def index():
    return "Backend is running. Use /api/start-journey from the frontend."


@app.route("/api/start-journey", methods=["POST"])
def start_journey():
    try:
        data = request.json or {}

        journey_data = {
            "journey_id": str(uuid.uuid4()),
            "vehicle_id": data["vehicleId"],
            "owner_phone": data["ownerPhone"],
            "owner_name": data["ownerName"],
            "predicted_issue": data["issue"],
            "risk_level": data["riskLevel"],
        }

        result = execute_maintenance_journey(journey_data)
        voice_script = result.get("voice_script", "Your maintenance journey has been processed.")

        audio_filename = generate_voice_audio(voice_script)
        base_url = request.host_url.rstrip("/")
        audio_url = f"{base_url}/audio/{audio_filename}"

        completed_journeys[journey_data["journey_id"]] = {
            "journey_id": journey_data["journey_id"],
            "vehicle_id": journey_data["vehicle_id"],
            "owner_name": journey_data["owner_name"],
            "owner_phone": journey_data["owner_phone"],
            "predicted_issue": journey_data["predicted_issue"],
            "risk_level": journey_data["risk_level"],
            "status": str(result["state"]),
            "booking_confirmed": bool(result["booking_confirmed"]),
            "declined": bool(result["declined"]),
            "completed_at": datetime.now().isoformat(),
            "voice_script": voice_script,
            "voice_audio_url": audio_url,
        }

        return jsonify(
            {
                "success": True,
                "journey_id": journey_data["journey_id"],
                "status": result["state"],
                "booking_confirmed": result["booking_confirmed"],
                "declined": result["declined"],
                "voice_script": voice_script,
                "voice_audio_url": audio_url,
            }
        )
    except Exception as e:
        print("ERROR in /api/start-journey:", e)
        return jsonify({"error": "Connection error."}), 500


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@app.route("/api/journey/<journey_id>", methods=["GET"])
def get_journey_status(journey_id):
    if journey_id in completed_journeys:
        return jsonify(completed_journeys[journey_id])
    elif journey_id in active_journeys:
        return jsonify({"status": "in_progress", "data": active_journeys[journey_id]})
    else:
        return jsonify({"error": "Journey not found"}), 404


@app.route("/api/journeys", methods=["GET"])
def list_journeys():
    return jsonify(list(completed_journeys.values()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
