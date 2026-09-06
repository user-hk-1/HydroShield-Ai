import os
import requests
import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

# --- Tool 1: Live Hourly Weather Telemetry ---
@tool
def check_hourly_weather(latitude: float, longitude: float) -> str:
    """
    Fetches CURRENT weather + HOURLY FORECAST for the specified GPS coordinates.
    """
    load_dotenv(override=True)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API Key not configured."

    try:
        # Current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5).json()

        # Hourly forecast (3-hour intervals)
        fc_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}&units=metric&cnt=3"
        fc_res = requests.get(fc_url, timeout=5).json()

        output = ""
        if "main" in res:
            temp = res['main']['temp']
            desc = res['weather'][0]['description']
            pressure = res['main']['pressure']
            humidity = res['main']['humidity']
            wind = res['wind']['speed']
            rain_1h = res.get('rain', {}).get('1h', 0)
            rain_3h = res.get('rain', {}).get('3h', 0)
            is_raining = rain_1h > 0 or rain_3h > 0 or any(w in desc.lower() for w in ['rain', 'drizzle', 'thunderstorm'])

            loc_name = res.get("name", "Unknown Sector")
            output += f"📍 Location: {loc_name} ({latitude:.4f}, {longitude:.4f})\n"
            output += f"🌡️ Weather: {desc.title()} | Temp: {temp}°C | Humidity: {humidity}% | Wind: {wind} m/s\n"
            output += f"🌧️ Rain: {max(rain_1h, rain_3h)}mm/hr | Active Rainfall: {'YES' if is_raining else 'NO'}\n"

        forecast_list = fc_res.get("list", [])
        if forecast_list:
            output += "📊 Forecast (next 6 hrs):\n"
            for fc in forecast_list:
                dt = datetime.datetime.fromtimestamp(fc['dt'])
                time_str = dt.strftime("%I:%M %p")
                f_desc = fc['weather'][0]['description']
                f_temp = fc['main']['temp']
                f_rain = fc.get('rain', {}).get('3h', 0)
                rain_flag = "🌧️ RAIN" if f_rain > 0 or 'rain' in f_desc.lower() else "☀️ DRY"
                output += f"  • {time_str}: {f_desc.title()}, {f_temp}°C [{rain_flag}]\n"

        return output if output else "Weather telemetry unavailable."
    except Exception as e:
        return f"Weather query error: {e}"


# --- Tool 2: Telegram Bot Real-Time Hazard Alert Dispatcher ---
@tool
def send_telegram_alert(message: str, lat: float, lon: float) -> str:
    """
    Dispatches a real-time severe hazard alert to municipal responders via Telegram Bot push notification.
    MUST be used if a severe hazard (like a waterlogged road or critical pothole under active rainfall) is confirmed.
    """
    load_dotenv(override=True)
    gmaps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    full_message = (
        f"🚨 *HYDROSHIELD SEVERE HAZARD ALERT* 🚨\n\n"
        f"⚠️ *Report:* {message}\n"
        f"📍 *GPS Location:* `{lat:.6f}, {lon:.6f}`\n\n"
        f"🗺️ [Open Navigation in Google Maps]({gmaps_link})"
    )

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    pushbullet_key = os.getenv("PUSHBULLET_API_KEY")

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            bot_token = bot_token or st.secrets.get("TELEGRAM_BOT_TOKEN")
            chat_id = chat_id or st.secrets.get("TELEGRAM_CHAT_ID")
            pushbullet_key = pushbullet_key or st.secrets.get("PUSHBULLET_API_KEY")
    except Exception:
        pass


    outputs = []

    if bot_token and chat_id and bot_token.strip() and chat_id.strip():
        try:
            tg_url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
            tg_res = requests.post(
                tg_url,
                json={
                    "chat_id": chat_id.strip(),
                    "text": full_message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": False
                },
                timeout=8
            )
            if tg_res.status_code == 200:
                outputs.append("✅ Telegram Bot: Instant hazard alert notification dispatched to your Telegram!")
            else:
                outputs.append(f"❌ Telegram API Error {tg_res.status_code}: {tg_res.text}")
        except Exception as e:
            outputs.append(f"❌ Telegram Network Error: {e}")
    else:
        outputs.append("⚠️ Telegram Bot Simulation: Alert formatted. (Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env for live Telegram alerts).")

    # Optional Pushbullet fallback if key is configured
    if pushbullet_key and pushbullet_key.strip():
        try:
            pb_res = requests.post(
                "https://api.pushbullet.com/v2/pushes",
                headers={"Access-Token": pushbullet_key.strip(), "Content-Type": "application/json"},
                json={
                    "type": "note",
                    "title": "🚨 HYDROSHIELD SEVERE HAZARD DETECTED",
                    "body": f"{message}\n\nNav: {gmaps_link}"
                },
                timeout=8
            )
            if pb_res.status_code == 200:
                outputs.append("✅ Pushbullet: Notification also sent to Pushbullet.")
        except Exception:
            pass

    return "\n".join(outputs)


@tool
def send_sms_alert(message: str, lat: float, lon: float) -> str:
    """
    SMS/Push alert tool wrapper that dispatches via Telegram Bot.
    """
    return send_telegram_alert(message, lat, lon)


# --- Tool 3: Municipal Contact Search ---
search_tool = DuckDuckGoSearchRun()

# --- ReAct Agent Prompt ---
REACT_SYSTEM_PROMPT = """You are the HydroShield-AI Backend Municipal Dispatcher.

Goal: You autonomously monitor CCTV classification events and cross-reference them with live weather telemetry. If a hazard is deemed SEVERE, you must dispatch an alert notification to municipal responders via Telegram.

STRICT PROTOCOL:
1. When you receive a hazard report, FIRST use `check_hourly_weather` at the provided coordinates.
2. Evaluate Severity:
   - If the hazard is 'Waterlogged Road' or 'Waterlogged road', it is always SEVERE.
   - If the hazard is 'Pothole Detected' or 'Pothole detect' AND there is 'Active Rainfall' (YES), it is SEVERE.
   - Otherwise, it is MODERATE/SAFE.
3. If SEVERE, you MUST autonomously use `send_telegram_alert` (or `send_sms_alert`). Keep the message concise (e.g., "Severe Waterlogging detected in Sector X. Immediate clearance required.").
4. Finally, output a brief internal log summarizing your actions:
   - "Alert Status: Dispatched via Telegram (or Not Required)"
   - "Reason: [Concise reason based on your evaluation]"
"""

def run_municipal_agent(hazard_prediction: str, sector: str, lat: float, lon: float) -> str:
    """
    Executes the agent to evaluate hazard severity and autonomously dispatch Telegram/SMS alerts.
    """
    load_dotenv(override=True)
    groq_api = os.getenv("GROQ_API_KEY")
    if not groq_api:
        return "⚠️ GROQ_API_KEY is missing from your .env file."

    llm = ChatGroq(model='openai/gpt-oss-120b', api_key=groq_api, temperature=0.1)
    tools = [check_hourly_weather, send_telegram_alert, send_sms_alert, search_tool]

    agent = create_react_agent(llm, tools=tools, prompt=REACT_SYSTEM_PROMPT)

    query = f"CCTV at {lat}, {lon} (Sector: '{sector}') detected: '{hazard_prediction}'. Evaluate severity against live weather and dispatch Telegram alert if required."

    try:
        inputs = {"messages": [("user", query)]}
        response = agent.invoke(inputs)
        return response["messages"][-1].content
    except Exception as e:
        return f"⚠️ Agent execution error: {e}"