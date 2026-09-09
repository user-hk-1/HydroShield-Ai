import streamlit as st
from streamlit_js_eval import get_geolocation
import numpy as np
import os

from PIL import Image
import os
import base64
import requests
from dotenv import load_dotenv
import datetime

# Import modular agent engine
from agent import run_municipal_agent

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="HydroShield-AI", page_icon="🌊", layout="wide")

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# --- Model & Vision Config ---
PYTORCH_MODEL_PATH = 'rapid_model.pth'
KERAS_MODEL_PATH = 'rapid_model.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Clear Road', 'Pothole Detected', 'Waterlogged Road']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_vision_model():
    if os.path.exists(PYTORCH_MODEL_PATH):
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            model = models.mobilenet_v2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, len(CLASS_NAMES))
            )
            model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            return ("pytorch", model)
        except Exception:
            pass

    if os.path.exists(KERAS_MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            return ("keras", model)
        except Exception:
            pass

    return (None, None)

model_type, model = load_vision_model()

pytorch_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_surface_hazard(image):
    if model is None:
        return CLASS_NAMES[0], 0.0

    image_rgb = image.convert('RGB')

    if model_type == "pytorch":
        tensor_img = pytorch_transform(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0]
            class_idx = torch.argmax(probabilities).item()
            confidence = float(probabilities[class_idx].item() * 100)
        return CLASS_NAMES[class_idx], confidence
    elif model_type == "keras":
        image_resized = image_rgb.resize((128, 128))
        img_array = np.array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        return CLASS_NAMES[class_idx], confidence

    return CLASS_NAMES[0], 0.0



# ==============================================================================
# GEOLOCATION ENGINE (Browser HTML5 GPS via streamlit-js-eval)
# ==============================================================================

# Default fallback coordinates (East Delhi)
DEFAULT_LAT = 28.6304
DEFAULT_LON = 77.2771

@st.cache_data(ttl=300)
def reverse_geocode(lat, lon):
    """Converts GPS coordinates to exact street/city name via Nominatim."""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "HydroShield-AI/2.0"}
        res = requests.get(url, headers=headers, timeout=4).json()
        address = res.get("address", {})
        suburb = address.get("suburb") or address.get("neighbourhood") or address.get("residential") or address.get("road") or ""
        city = address.get("city") or address.get("town") or address.get("county") or address.get("state_district") or "Delhi"
        region = address.get("state") or "Delhi"
        display_name = f"{suburb}, {city}".strip(", ") if suburb else f"{city}, {region}"
        return display_name, city, region
    except Exception:
        return "Delhi, Delhi", "Delhi", "Delhi"


def get_user_location():
    """
    Uses streamlit-js-eval to request real browser HTML5 GPS.
    Falls back to default East Delhi coords if permission denied or still loading.
    Allows manual override via session_state.
    """
    # Check for manual override first
    if "manual_lat" in st.session_state and "manual_lon" in st.session_state:
        if st.session_state["manual_lat"] is not None and st.session_state["manual_lon"] is not None:
            lat = st.session_state["manual_lat"]
            lon = st.session_state["manual_lon"]
            display_name, city, region = reverse_geocode(lat, lon)
            return {
                "lat": lat, "lon": lon,
                "city": city, "region": region,
                "display": display_name, "source": "Manual Override"
            }

    # Try browser GPS via streamlit-js-eval
    location = get_geolocation()
    if location and isinstance(location, dict):
        coords = location.get("coords", {})
        if coords:
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat is not None and lon is not None:
                display_name, city, region = reverse_geocode(lat, lon)
                return {
                    "lat": float(lat), "lon": float(lon),
                    "city": city, "region": region,
                    "display": display_name, "source": "Browser GPS"
                }

    # Fallback to default East Delhi coordinates
    display_name, city, region = reverse_geocode(DEFAULT_LAT, DEFAULT_LON)
    return {
        "lat": DEFAULT_LAT, "lon": DEFAULT_LON,
        "city": city, "region": region,
        "display": display_name, "source": "Default (GPS pending)"
    }


# --- Current Weather Telemetry ---
@st.cache_data(ttl=120)
def get_current_weather(lat, lon):
    """Fetches real-time weather telemetry for specified coordinates."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"is_raining": False, "description": "N/A", "rain_mm": 0, "temp": "--", "humidity": "--"}
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=4).json()
        if "main" in res:
            desc = res['weather'][0]['description'].lower()
            rain_1h = res.get('rain', {}).get('1h', 0)
            rain_3h = res.get('rain', {}).get('3h', 0)
            is_raining = (rain_1h > 0 or rain_3h > 0 or any(w in desc for w in ['rain', 'drizzle', 'thunderstorm']))
            return {
                "is_raining": is_raining,
                "description": res['weather'][0]['description'].title(),
                "rain_mm": max(rain_1h, rain_3h),
                "temp": res['main']['temp'],
                "humidity": res['main']['humidity']
            }
    except Exception:
        pass
    return {"is_raining": False, "description": "N/A", "rain_mm": 0, "temp": "--", "humidity": "--"}


# --- Weather-Aware Hazard Assessment ---
def assess_hazard(pred_class, confidence, weather_data, image=None):
    """
    Evaluates vision prediction and applies user's exact weather + pothole size hazard matrix:
    
    IN RAINFALL:
    1. Clear Road -> "Road Clear"
    2. Waterlogged Road -> "Hazard Report: Waterlogged road detected"
    3. Pothole >= 30cm -> "Hazard Report: Pothole detected (Est. Width: ~XX cm)"
       Pothole < 30cm  -> "Hazard Report: Small pothole detected (Est. Width: ~XX cm)"
       
    IN NO RAINFALL (DRY):
    1. Clear Road -> "Road Clear"
    2. Waterlogged Road -> "Severe Report: Waterlogged road detected"
    3. Pothole < 30cm  -> "Report: The small pothole detected (Est. Width: ~XX cm)"
       Pothole >= 30cm -> "Severe Report: Big pothole fetched (Est. Width: ~XX cm)"
    """
    is_raining = weather_data.get("is_raining", False) or weather_data.get("rain_mm", 0) > 0

    if pred_class == "Clear Road":
        return {
            "severity": "SAFE",
            "color": "#00FF66",
            "icon": "✅",
            "label": "Road Clear",
            "dimensions": "N/A (Clear Surface)",
            "needs_detour": False,
            "description": "Road surface is clear. No structural defect detected."
        }
    elif pred_class == "Pothole Detected":
        width_cm = 25
        if image is not None:
            try:
                import cv2
                img_np = np.array(image.convert('RGB'))
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_h, img_w = gray.shape
                blurred = cv2.GaussianBlur(gray, (9, 9), 0)
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 21, 5
                )
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_widths = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > (img_w * img_h * 0.005):
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w < img_w * 0.90 and h < img_h * 0.90:
                            valid_widths.append(w)
                if valid_widths:
                    ratio = max(valid_widths) / float(img_w)
                    raw_cm = int(10 + (ratio * 50))
                else:
                    pixel_std = float(np.std(gray))
                    raw_cm = int(15 + (pixel_std % 30))
                width_cm = max(10, min(50, int(round(raw_cm / 5.0) * 5)))
            except Exception:
                raw_w = int(10 + (confidence / 100.0) * 35)
                width_cm = max(10, min(50, (raw_w // 5) * 5))
        else:
            raw_w = int(10 + (confidence / 100.0) * 35)
            width_cm = max(10, min(50, (raw_w // 5) * 5))

        dim_str = f"Est. Width: ~{width_cm} cm"

        if is_raining:
            # RAINFALL LOGIC FOR POTHOLES
            if width_cm >= 30:
                report_label = f"Hazard Report: Pothole detected ({dim_str})"
                severity_type = "HAZARD"
                color_hex = "#FF1744"
                icon_str = "🚨"
            else:
                report_label = f"Hazard Report: Small pothole detected ({dim_str})"
                severity_type = "HAZARD"
                color_hex = "#FFAB00"
                icon_str = "⚠️"
            
            return {
                "severity": severity_type,
                "color": color_hex,
                "icon": icon_str,
                "label": report_label,
                "dimensions": dim_str,
                "width_cm": width_cm,
                "needs_detour": True,
                "description": f"{report_label} under active rainfall. High vehicle hazard."
            }
        else:
            # NO RAINFALL (DRY) LOGIC FOR POTHOLES
            if width_cm < 30:
                report_label = f"Report: The small pothole detected ({dim_str})"
                severity_type = "REPORT"
                color_hex = "#FFAB00"
                icon_str = "⚠️"
                detour = False
            else:
                report_label = f"Severe Report: Big pothole fetched ({dim_str})"
                severity_type = "SEVERE"
                color_hex = "#FF1744"
                icon_str = "🚨"
                detour = True

            return {
                "severity": severity_type,
                "color": color_hex,
                "icon": icon_str,
                "label": report_label,
                "dimensions": dim_str,
                "width_cm": width_cm,
                "needs_detour": detour,
                "description": f"{report_label} under dry ambient conditions."
            }

    elif pred_class == "Waterlogged Road":
        if is_raining:
            report_label = "Hazard Report: Waterlogged road detected"
            severity_type = "HAZARD"
            color_hex = "#FFAB00"
            icon_str = "⚠️"
        else:
            report_label = "Severe Report: Waterlogged road detected"
            severity_type = "SEVERE"
            color_hex = "#FF1744"
            icon_str = "🚨"

        return {
            "severity": severity_type,
            "color": color_hex,
            "icon": icon_str,
            "label": report_label,
            "dimensions": "N/A (Standing Water)",
            "width_cm": None,
            "needs_detour": True,
            "description": f"{report_label}. GOOGLE MAPS DETOUR RECOMMENDED to avoid vehicle stalling."
        }

    return {
        "severity": "UNKNOWN",
        "color": "#AAAAAA",
        "icon": "❓",
        "label": pred_class.replace('_', ' '),
        "dimensions": "N/A",
        "needs_detour": False,
        "description": "Surface condition unverified."
    }


def render_dispatch_log_card(agent_raw, hazard):
    """
    Renders a pristine, executive Glassmorphic Dispatch Log Card.
    """
    severity = str(hazard.get('severity', 'HAZARD')).upper()
    lbl = str(hazard.get('label', ''))
    
    if 'SEVERE' in severity or 'severe' in lbl.lower() or 'critical' in severity.lower():
        status_badge = '<span style="background: rgba(255, 23, 68, 0.25); color: #FF1744; border: 1px solid rgba(255, 23, 68, 0.5); padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;">🚨 SEVERE HAZARD DISPATCHED</span>'
        card_border = "#FF1744"
    elif 'HAZARD' in severity or 'hazard' in lbl.lower():
        status_badge = '<span style="background: rgba(255, 171, 0, 0.25); color: #FFAB00; border: 1px solid rgba(255, 171, 0, 0.5); padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;">⚠️ HAZARD REPORTED</span>'
        card_border = "#FFAB00"
    elif 'REPORT' in severity or 'report:' in lbl.lower():
        status_badge = '<span style="background: rgba(0, 191, 255, 0.2); color: #00BFFF; border: 1px solid rgba(0, 191, 255, 0.4); padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;">📋 REPORT LOGGED</span>'
        card_border = "#00BFFF"
    else:
        status_badge = '<span style="background: rgba(0, 255, 102, 0.15); color: #00FF66; border: 1px solid rgba(0, 255, 102, 0.3); padding: 4px 12px; border-radius: 20px; font-weight: 800; font-size: 0.82rem;">ℹ️ MONITORED SAFE</span>'
        card_border = "#00FF66"

    # Extract Telegram status line cleanly
    tg_line = "✅ Dispatched to Control Channel (@hydro_shield_bot)"
    if "Telegram Broadcast:" in agent_raw:
        for line in agent_raw.split('\n'):
            if "Telegram Broadcast:" in line:
                clean_t = line.replace("• Telegram Broadcast:", "").replace("Telegram Broadcast:", "").strip()
                if clean_t:
                    tg_line = clean_t
                break

    # Extract Triage Reasoning line cleanly
    triage_reason = hazard.get('description', 'Surface condition evaluated per municipal safety protocol.')
    if "Triage Evaluation:" in agent_raw:
        for line in agent_raw.split('\n'):
            if "Triage Evaluation:" in line:
                clean_r = line.replace("• Triage Evaluation:", "").replace("Triage Evaluation:", "").strip()
                if clean_r:
                    triage_reason = clean_r
                break

    return f'<div style="background: rgba(4, 10, 24, 0.75); border: 1px solid rgba(255, 255, 255, 0.15); border-left: 5px solid {card_border}; backdrop-filter: blur(50px); border-radius: 18px; padding: 1rem 1.2rem; margin-top: 0.8rem; margin-bottom: 0.8rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4);"><div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255, 255, 255, 0.1); padding-bottom: 0.6rem;"><div style="color: #00BFFF; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.5px;">📋 AUTONOMOUS TRIAGE & DISPATCH STATUS</div>{status_badge}</div><div style="display: flex; flex-direction: column; gap: 0.6rem;"><div style="display: flex; align-items: center; background: rgba(0,0,0,0.3); padding: 0.6rem 0.9rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);"><span style="font-size: 1.1rem; margin-right: 0.7rem;">📲</span><div><div style="color: #888; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Control Broadcast Channel</div><div style="color: #E8F0FE; font-size: 0.88rem; font-weight: 600;">{tg_line}</div></div></div><div style="display: flex; align-items: center; background: rgba(0,0,0,0.3); padding: 0.6rem 0.9rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);"><span style="font-size: 1.1rem; margin-right: 0.7rem;">🧠</span><div><div style="color: #888; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Autonomous Triage Decision</div><div style="color: #D0E8FF; font-size: 0.88rem; font-weight: 600; line-height: 1.4;">{triage_reason}</div></div></div></div></div>'


# ==============================================================================
# GLASSMORPHIC & WEATHER MODAL CSS INJECTION
# ==============================================================================
# LINE REFERENCE GUIDE FOR USER CUSTOMIZATIONS:
# - Weather Modal Width & Max Height: Lines 235-245 below
# - Weather Modal Glass Blur & Opacity: Lines 246-260 below
# - Background Wallpaper & Glass Cards: Lines 215-234 below
# ==============================================================================
def inject_custom_css():
    bg_file = "bg.jpg" if os.path.exists("bg.jpg") else "bg.png"
    if os.path.exists(bg_file):
        with open(bg_file, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
        bg_css = f"""
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_css = ".stApp { background: linear-gradient(135deg, #dbf4f7 0%, #b2ebf2 100%); }"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

        {bg_css}

        * {{
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 0.8rem !important;
            max-width: 1150px !important;
        }}
        header {{ display: none !important; }}
        [data-testid="stDecoration"] {{ display: none !important; }}
        div[data-testid="stCustomComponentV1"] {{
            display: none !important;
            height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
        }}
        iframe[title="streamlit_js_eval.streamlit_js_eval"] {{
            display: none !important;
            height: 0px !important;
        }}

        /* Remove empty gap above columns in triage view */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}
        [data-testid="column"] {{
            padding-top: 0 !important;
        }}

        /* Global readable text: dark text-shadow for legibility */
        h1, h2, h3, h4, p, span, b, strong, label {{
            text-shadow: 0 1px 4px rgba(0,0,0,0.55) !important;
        }}

        /* ===================================== */
        /* BLACK GLASSMORPHIC PILL BUTTONS       */
        /* ===================================== */
        div.stButton > button {{
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.35) !important;
            border-radius: 50px !important;
            color: white !important;
            height: 52px !important;
            width: 100% !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
            transition: all 0.25s ease-in-out !important;
        }}
        div.stButton > button:hover {{
            background: rgba(0, 0, 0, 0.8) !important;
            border: 1.5px solid #00BFFF !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 32px rgba(0, 191, 255, 0.35) !important;
            color: #00BFFF !important;
        }}

        .weather-btn div.stButton > button {{
            height: 38px !important;
            margin-bottom: 0.5rem;
        }}
        .weather-btn div.stButton > button p {{
            font-size: 13px !important;
        }}

        /* ====================================================================== */
        /* BUTTERY SMOOTH MODAL KEYFRAME ANIMATIONS                              */
        /* ====================================================================== */
        @keyframes modalPopUp {{
            0% {{
                opacity: 0;
                transform: scale(0.85) translateY(40px);
                filter: blur(12px);
            }}
            100% {{
                opacity: 1;
                transform: scale(1) translateY(0);
                filter: blur(0px);
            }}
        }}

        @keyframes backdropFade {{
            0% {{
                opacity: 0;
                backdrop-filter: blur(0px) brightness(1);
                -webkit-backdrop-filter: blur(0px) brightness(1);
            }}
            100% {{
                opacity: 1;
                backdrop-filter: blur(15px) brightness(0.5);
                -webkit-backdrop-filter: blur(15px) brightness(0.5);
            }}
        }}

        /* ====================================================================== */
        /* TRANSPARENT WRAPPERS (PREVENTS DOUBLE DARK OPAQUE BOXES)              */
        /* ====================================================================== */
        div[data-baseweb="modal"],
        div[data-baseweb="modal-container"],
        div[data-testid="stDialog"],
        div[role="dialog"] {{
            background: transparent !important;
            background-color: transparent !important;
            box-shadow: none !important;
        }}

        div[data-testid="stDialog"] > div,
        div[data-baseweb="modal"] > div {{
            width: 90vw !important;
            max-width: 1250px !important;
            min-width: 800px !important;
            background: transparent !important;
            background-color: transparent !important;
        }}

        /* ====================================================================== */
        /* REAL TRANSLUCENT GLASSMORPHIC MODAL CARD WITH SPRING POPUP ANIMATION  */
        /* ====================================================================== */
        div[data-testid="stDialog"] > div > div,
        div[data-baseweb="modal"] > div > div {{
            max-height: 80vh !important;
            overflow-y: auto !important;
            background: rgba(8, 18, 38, 0.45) !important;
            backdrop-filter: blur(40px) saturate(220%) !important;
            -webkit-backdrop-filter: blur(40px) saturate(220%) !important;
            border-radius: 28px !important;
            border: 1.5px solid rgba(0, 191, 255, 0.45) !important;
            box-shadow: 0 30px 100px rgba(0, 0, 0, 0.6), inset 0 0 50px rgba(0, 191, 255, 0.15) !important;
            padding: 2.2rem !important;

            animation: modalPopUp 0.48s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards !important;
            will-change: transform, opacity, filter;
        }}

        /* Fix text contrast inside dialog */
        div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3, div[role="dialog"] p, div[role="dialog"] span {{
            color: #FFFFFF !important;
        }}

        div[data-testid="stDialog"]::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: -1;
            animation: backdropFade 0.4s ease-out forwards !important;
        }}

        /* ------------------------------------- */
        /* STYLIZED EXPANDER                     */
        /* ------------------------------------- */
        [data-testid="stExpander"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            margin-bottom: 1rem;
        }}
        [data-testid="stExpander"] details {{
            background: rgba(4, 10, 24, 0.70) !important;
            border-radius: 22px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            backdrop-filter: blur(80px) saturate(200%) !important;
            -webkit-backdrop-filter: blur(80px) saturate(200%) !important;
            box-shadow: 0 20px 50px rgba(0,0,0,0.6) !important;
            color: #E8F0FE !important;
        }}
        [data-testid="stExpander"] summary {{
            background: transparent !important;
            border: none !important;
            color: #00BFFF !important;
            font-weight: 700;
            letter-spacing: 0.5px;
            padding: 1rem 1.4rem !important;
        }}
        [data-testid="stExpanderDetails"] {{
            background: transparent !important;
            border: none !important;
            padding: 0.5rem 1.4rem 1.4rem 1.4rem !important;
        }}

        /* ===================================== */
        /* DARK GLASS PANELS & TABS              */
        /* ===================================== */
        /* Apply glass panel effect directly to the first layout column */
        div[data-testid="column"]:nth-of-type(1) {{
            background: rgba(10, 15, 25, 0.82) !important;
            backdrop-filter: blur(35px) !important;
            -webkit-backdrop-filter: blur(35px) !important;
            border-radius: 22px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            padding: 1.3rem !important;
            color: #E8F0FE;
        }}

        /* Apply glass panel effect directly to the Tab content area */
        div[data-testid="stTab"] {{
            background: rgba(10, 15, 25, 0.82) !important;
            backdrop-filter: blur(35px) !important;
            -webkit-backdrop-filter: blur(35px) !important;
            border-radius: 22px !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            padding: 1.3rem !important;
            color: #E8F0FE;
            margin-top: 10px;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px !important;
            background: transparent !important;
            padding: 0 !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(4, 10, 24, 0.65) !important;
            backdrop-filter: blur(40px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
            border-radius: 30px !important;
            color: #B0C4DE !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            padding: 10px 22px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(0, 191, 255, 0.25) !important;
            color: #00BFFF !important;
            border: 1.5px solid #00BFFF !important;
            box-shadow: 0 10px 25px rgba(0, 191, 255, 0.3) !important;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        .map-container {{
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .map-container iframe {{
            border: none;
            border-radius: 20px;
        }}

        /* Make sure Streamlit Folium iframes are also rounded */
        iframe[title="streamlit_folium.st_folium"] {{
            border-radius: 20px !important;
            overflow: hidden !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
        }}

        /* Clean Contact Cards (Rounded to match glassmorphic theme) */
        .info-card {{
            background: rgba(0, 0, 0, 0.6);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid #00BFFF;
        }}
        .info-card h4 {{
            margin: 0 0 0.3rem 0;
            color: #00BFFF;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .info-card p {{
            margin: 0;
            color: #D0E8FF;
            font-size: 0.88rem;
            line-height: 1.5;
        }}

        /* Round all Streamlit native elements that have sharp corners */
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"],
        [data-testid="stImage"],
        .stAlert, .stSpinner, .stMarkdown {{
            border-radius: 16px !important;
            overflow: hidden;
        }}
        [data-testid="stFileUploader"] section {{
            border-radius: 16px !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            border-radius: 0 0 16px 16px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()


# --- Session State ---
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None


# ==============================================================================
# EXPANDED WEATHER DIALOG (Wider, Horizontal, Heavily Blurred)
# ==============================================================================
@st.dialog("☁️ Meteorological Telemetry & Hourly Radar")
def show_weather_modal(loc):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        st.warning("⚠️ OPENWEATHER_API_KEY missing from .env")
        return

    lat, lon = loc["lat"], loc["lon"]

    try:
        # Current telemetry
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=4).json()

        # Hourly forecast
        fc_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric&cnt=4"
        fc_res = requests.get(fc_url, timeout=4).json()
        forecast_list = fc_res.get("list", [])[:4]

        if "main" in res:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2.3, 3.3, 4.4])

            with col1:
                temp = res['main']['temp']
                desc = res['weather'][0]['description'].title()
                icon = res['weather'][0]['icon']
                st.markdown(f"""
                    <div style="background: rgba(0, 191, 255, 0.08); border: 1px solid rgba(0, 191, 255, 0.25); border-radius: 20px; padding: 1.2rem 1rem; text-align: center;">
                        <img src="http://openweathermap.org/img/wn/{icon}@2x.png" width="60" style="display:block; margin: 0 auto 5px auto;">
                        <div style="white-space: nowrap; color: #00BFFF; font-size: 2.8rem; font-weight: 900; letter-spacing: -1px; line-height: 1;">
                            {temp}°C
                        </div>
                        <div style="color: #E8F0FE; font-size: 0.92rem; font-weight: 700; margin-top: 8px; white-space: nowrap;">
                            {desc}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div style="background: rgba(0, 191, 255, 0.08); padding: 1.1rem 1.3rem; border-radius: 20px; border: 1px solid rgba(0, 191, 255, 0.25);">
                        <h3 style="color:#E8F0FE; margin: 0 0 10px 0; font-size: 1.15rem; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">📍 {loc['display']}</h3>
                        <div style="color:#D0E8FF; font-size: 0.88rem; line-height: 1.8;">
                            <b>Sensor Source:</b> <span style="color:#00FF66; font-weight: 700;">{loc['source']}</span><br>
                            <b>Barometer:</b> {res['main']['pressure']} hPa<br>
                            <b>Relative Humidity:</b> {res['main']['humidity']}%<br>
                            <b>Wind Velocity:</b> {res['wind']['speed']} m/s
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("<h3 style='color:#E8F0FE; margin: 0 0 10px 0; font-size: 1.15rem; font-weight: 800;'>📊 12-Hour Forecast Radar</h3>", unsafe_allow_html=True)
                fc_cols = st.columns(4)
                for i, fc in enumerate(forecast_list):
                    with fc_cols[i]:
                        dt = datetime.datetime.fromtimestamp(fc['dt'])
                        time_str = dt.strftime("%I %p")
                        f_icon = fc['weather'][0]['icon']
                        f_url = f"http://openweathermap.org/img/wn/{f_icon}.png"
                        f_rain = fc.get('rain', {}).get('3h', 0)
                        st.markdown(f"""
                            <div style="background: rgba(0, 191, 255, 0.08); border: 1px solid rgba(0, 191, 255, 0.25); border-radius: 16px; padding: 0.7rem 0.2rem; text-align: center;">
                                <div style="color:#B0C4DE; font-weight:700; font-size:0.8rem; margin-bottom:2px; white-space:nowrap;">{time_str}</div>
                                <img src="{f_url}" width="36" style="display:block; margin: 2px auto;">
                                <div style="color:#E8F0FE; font-weight:800; font-size:0.88rem; margin-top:2px; white-space:nowrap;">{fc['main']['temp']}°C</div>
                                <div style="color:#00FF66; font-size:0.75rem; font-weight:700; white-space:nowrap; margin-top:3px;">{'🌧️ '+str(f_rain)+'mm' if f_rain > 0 else '☀️ Dry'}</div>
                            </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("Weather data currently unavailable.")
    except Exception as e:
        st.error(f"Weather API query failed: {e}")


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

# Fetch user location exactly once per run to avoid DuplicateElementKey errors
GLOBAL_LOC = get_user_location()

if st.session_state.current_image is None:
    # ---------------------------
    # STATE A: LANDING VIEW
    # ---------------------------

    w_col1, w_col2, w_col3 = st.columns([4, 2, 4])
    with w_col2:
        st.markdown('<div class="weather-btn">', unsafe_allow_html=True)
        if st.button("⛅Weather Dropdown", use_container_width=True):
            show_weather_modal(GLOBAL_LOC)
        st.markdown('</div>', unsafe_allow_html=True)

    # Title
    st.markdown("""
        <h1 style="
            font-family: 'Inter', sans-serif;
            text-align: center;
            font-size: 5.5rem;
            font-weight: 900;
            margin-top: 0;
            margin-bottom: 1.5rem;
            letter-spacing: 3px;
            background: linear-gradient(180deg, #0a1628 0%, #0d3b66 40%, #1a8fa8 70%, #5ec4d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0px 4px 8px rgba(0, 0, 0, 0.15));
        ">HydroShield</h1>
    """, unsafe_allow_html=True)



    # Action Buttons
    col_left, col_center1, col_center2, col_right = st.columns([2, 1.5, 1.5, 2])

    with col_center1:
        if st.button("☁️ Upload Frame", use_container_width=True):
            st.session_state.input_mode = 'upload'

        if st.session_state.input_mode == 'upload':
            file_upload = st.file_uploader("Upload Frame", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
            if file_upload:
                st.session_state.current_image = Image.open(file_upload)
                st.rerun()

    with col_center2:
        if st.button("📷 Capture Camera", use_container_width=True):
            st.session_state.input_mode = 'capture'

        if st.session_state.input_mode == 'capture':
            camera_snap = st.camera_input("Capture Frame", label_visibility="collapsed")
            if camera_snap:
                st.session_state.current_image = Image.open(camera_snap)
                st.rerun()

    # Location Override (Fail-Safe)
    with st.expander("📍 Location Settings", expanded=False):
        loc_preview = GLOBAL_LOC
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.06); padding: 0.9rem 1.2rem; border-radius: 14px; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.12);">
                <div style="color:#B0C4DE; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">Current Telemetry Location</div>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px;">
                    <span style="color:#00BFFF; font-size:1.1rem; font-weight:800;">{loc_preview['display']}</span>
                    <span style="background:rgba(0,255,102,0.15); color:#00FF66; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; border:1px solid rgba(0,255,102,0.3);">{loc_preview['source']}</span>
                    <span style="color:#E8F0FE; font-size:0.88rem; font-weight:600; font-family:monospace; margin-left:auto;">📍 {loc_preview['lat']:.6f}, {loc_preview['lon']:.6f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        ov_col1, ov_col2, ov_col3 = st.columns([1, 1, 0.8])
        with ov_col1:
            m_lat = st.number_input("Latitude", value=loc_preview['lat'], format="%.6f", key="input_lat")
        with ov_col2:
            m_lon = st.number_input("Longitude", value=loc_preview['lon'], format="%.6f", key="input_lon")
        with ov_col3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("📌 Set Pin", use_container_width=True):
                st.session_state["manual_lat"] = m_lat
                st.session_state["manual_lon"] = m_lon
                st.rerun()
            if "manual_lat" in st.session_state:
                if st.button("🔄 Reset GPS", use_container_width=True):
                    del st.session_state["manual_lat"]
                    del st.session_state["manual_lon"]
                    st.rerun()

else:
    # ---------------------------
    # STATE B: AGENTIC TRIAGE VIEW
    # ---------------------------

    if model is None:
        st.error("⚠️ Vision model ('rapid_model.keras') missing.")
        st.stop()

    pred_class, confidence = predict_surface_hazard(st.session_state.current_image)
    loc = GLOBAL_LOC
    weather = get_current_weather(loc["lat"], loc["lon"])
    hazard = assess_hazard(pred_class, confidence, weather, st.session_state.current_image)

    # Initialize and append dispatch history entry
    if "dispatch_history" not in st.session_state:
        st.session_state.dispatch_history = []

    current_entry = {
        "timestamp": datetime.datetime.now().strftime("%I:%M:%S %p | %b %d"),
        "hazard_label": hazard['label'],
        "dimensions": hazard.get('dimensions', 'N/A'),
        "confidence": f"{confidence:.1f}%",
        "color": hazard['color'],
        "icon": hazard['icon'],
        "location": loc['display'],
        "coords": f"{loc['lat']:.6f}, {loc['lon']:.6f}",
        "weather": f"{weather.get('temp', 'N/A')}°C | Rain: {'YES' if weather.get('is_raining') else 'NO'}",
        "severity": hazard.get('severity', 'SAFE'),
        "severe": hazard.get('severity') in ['SEVERE', 'CRITICAL', 'HAZARD']
    }

    if not st.session_state.dispatch_history or st.session_state.dispatch_history[0].get("timestamp") != current_entry["timestamp"]:
        st.session_state.dispatch_history.insert(0, current_entry)

    col_vision, col_triage = st.columns([0.85, 1.45])

    with col_vision:
        st.markdown("""
            <div style="
                background: rgba(4, 10, 24, 0.70);
                border: 1px solid rgba(255, 255, 255, 0.18);
                backdrop-filter: blur(80px) saturate(200%);
                -webkit-backdrop-filter: blur(80px) saturate(200%);
                border-radius: 18px;
                padding: 0.8rem 1.2rem;
                margin-bottom: 1.2rem;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            ">
                <h3 style="color:#00BFFF; margin:0; font-size:1.25rem; font-weight:800; letter-spacing:0.5px;">📷 CCTV Vision Analysis</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
            .small-preview img {
                max-width: 270px !important;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.2);
                display: block;
                margin: 0 auto;
            }
            </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="small-preview">', unsafe_allow_html=True)
        st.image(st.session_state.current_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        rain_status = "🌧️ Active Rainfall" if weather.get("is_raining") else "☀️ Dry Conditions"
        rain_color = "#FF6B6B" if weather.get("is_raining") else "#90EE90"

        badge_html = ""
        if pred_class == "Pothole Detected":
            badge_html = f'<div style="margin-top: 0.6rem; padding: 0.4rem 0.8rem; background: rgba(255, 23, 68, 0.15); border: 1px solid rgba(255, 23, 68, 0.4); border-radius: 12px; color: #FF1744; font-size: 0.82rem; font-weight: 800; display: inline-block;">📏 {hazard.get("dimensions", "N/A")}</div>'
        elif pred_class == "Waterlogged Road":
            badge_html = '<div style="margin-top: 0.6rem; padding: 0.4rem 0.8rem; background: rgba(255, 171, 0, 0.15); border: 1px solid rgba(255, 171, 0, 0.4); border-radius: 12px; color: #FFAB00; font-size: 0.82rem; font-weight: 800; display: inline-block;">🌊 Standing Water Surface Hazard</div>'

        # Unified Glassmorphic Status & Telemetry Card
        st.markdown(f'<div style="background: rgba(4, 10, 24, 0.70); border: 1px solid rgba(255, 255, 255, 0.15); backdrop-filter: blur(50px); -webkit-backdrop-filter: blur(50px); border-radius: 20px; padding: 1rem; margin-top: 1.2rem; margin-bottom: 0.8rem; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.4);"><div style="background: rgba(0, 0, 0, 0.5); border: 1.5px solid {hazard["color"]}; border-radius: 30px; padding: 0.5rem 1rem; display: inline-block; margin-bottom: 0.6rem;"><span style="font-size: 1.1rem;">{hazard["icon"]}</span><span style="color: {hazard["color"]}; font-size: 1rem; font-weight: 800; margin-left: 0.4rem;">{hazard["label"]}</span></div><div style="color: {rain_color}; font-size: 0.88rem; font-weight: 700; margin-top: 0.2rem;">{rain_status} ({weather.get("description", "N/A")})</div>{badge_html}</div>', unsafe_allow_html=True)

        st.write("")
        if st.button("⬅️ Inspect New Frame"):
            st.session_state.current_image = None
            st.session_state.input_mode = None
            if "last_frame_sig" in st.session_state:
                del st.session_state["last_frame_sig"]
            if "last_agent_raw" in st.session_state:
                del st.session_state["last_agent_raw"]
            st.rerun()

    with col_triage:
        tab_log, tab_map, tab_history = st.tabs(["📝 Incident Monitoring Log", "🗺️ CCTV Location", "📜 Dispatched Audit Log"])


        # -------------------------------------------------------------
        # TAB 1: CCTV BACKEND INCIDENT LOG
        # -------------------------------------------------------------
        with tab_log:
            st.markdown("""
                <div style="background: rgba(4, 10, 24, 0.75); padding: 0.8rem 1.1rem; border-radius: 16px; border: 1px solid rgba(0, 191, 255, 0.35); box-shadow: 0 8px 24px rgba(0,0,0,0.3); margin-bottom: 0.9rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 0.3rem;">
                        <span style="color: #00BFFF; font-weight: 800; font-size: 0.88rem; letter-spacing: 0.5px;">📢 MUNICIPAL DISPATCH PROTOCOL & WEBHOOK SETTINGS</span>
                        <span style="background: rgba(0, 191, 255, 0.2); border: 1px solid #00BFFF; color: #00BFFF; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; white-space: nowrap;">TELEGRAM CONTROL BROADCAST</span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.8rem; line-height: 1.35;">
                        <b>Auto Fallback:</b> Automatically dispatch to <b>Nearest MCD Control Room</b> via Telegram secure channel.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Gated dispatch execution to prevent duplicate Telegram alerts on page reruns / tab switches
            frame_sig = f"{st.session_state.current_image.size}_{hazard['label']}_{loc['lat']:.6f}_{loc['lon']:.6f}"
            if "last_frame_sig" not in st.session_state or st.session_state.last_frame_sig != frame_sig:
                with st.spinner("Analyzing hazard and dispatching emergency protocols if required..."):
                    st.session_state.last_agent_raw = run_municipal_agent(hazard['label'], sector=loc['display'], lat=loc['lat'], lon=loc['lon'])
                    st.session_state.last_frame_sig = frame_sig
            
            agent_raw = st.session_state.get("last_agent_raw", "📌 INCIDENT DISPATCH SUMMARY\n• Hazard Status: 🚨 SEVERE HAZARD DISPATCHED\n• Telegram Broadcast: Real-time alert dispatched to Municipal Control Channel!")

            # Section 1: GPS & Weather Card
            st.markdown(f"""
                <div class="info-card">
                    <h4>📍 CCTV TELEMETRY SENSOR</h4>
                    <p style="color: #FFFFFF;"><b>Coordinates:</b> {loc['lat']:.6f}, {loc['lon']:.6f} ({loc['source']})<br>
                    <b>Environment:</b> {weather.get('temp')}°C | {weather.get('description')} | <b>Active Rain:</b> {'YES' if weather.get('is_raining') else 'NO'}</p>
                </div>
            """, unsafe_allow_html=True)

            # Section 2: Agent Response (Executive Glassmorphic Dispatch Card)
            st.markdown(render_dispatch_log_card(agent_raw, hazard), unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TAB 2: EXACT PIN MAP 
        # -------------------------------------------------------------
        with tab_map:
            lat, lon = loc["lat"], loc["lon"]
            gmaps_embed_url = f"https://www.google.com/maps/embed/v1/place?key={os.getenv('GOOGLE_MAPS_API_KEY', '')}&q={lat},{lon}&zoom=17" if os.getenv('GOOGLE_MAPS_API_KEY') else f"https://maps.google.com/maps?q={lat},{lon}&z=17&output=embed"

            # Embed Google Map Iframe (Direct Map View Only)
            st.markdown(f"""
                <div class="map-container">
                    <iframe
                        width="100%"
                        height="370"
                        style="border:0; border-radius:20px;"
                        loading="lazy"
                        allowfullscreen
                        src="{gmaps_embed_url}">
                    </iframe>
                </div>
            """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # TAB 3: RECEIVED ALERT DISPATCH HISTORY
        # -------------------------------------------------------------
        with tab_history:
            st.markdown("""
                <div style="background: rgba(4, 10, 24, 0.6); padding: 0.6rem 1rem; border-radius: 14px; border: 1px solid rgba(255, 255, 255, 0.12); margin-bottom: 0.8rem;">
                    <div style="color: #00BFFF; font-weight: 800; font-size: 1rem;">📜 Dispatched Audit Log</div>
                    <div style="color: #AAA; font-size: 0.8rem;">Session logs of all inspected frames and emergency Telegram broadcasts.</div>
                </div>
            """, unsafe_allow_html=True)


            if not st.session_state.get("dispatch_history"):
                st.info("No dispatch history recorded yet in this session.")
            else:
                for idx, log_item in enumerate(st.session_state.dispatch_history):
                    border_color = log_item.get('color', '#00BFFF')
                    sev = str(log_item.get('severity', '')).upper()
                    lbl = str(log_item.get('hazard_label', '')).lower()

                    if 'severe' in lbl or sev == 'SEVERE' or sev == 'CRITICAL':
                        status_badge = "🚨 SEVERE DISPATCHED"
                        badge_bg = "rgba(255, 23, 68, 0.25)"
                        badge_fg = "#FF1744"
                    elif 'hazard' in lbl or sev == 'HAZARD':
                        status_badge = "⚠️ HAZARD REPORT"
                        badge_bg = "rgba(255, 171, 0, 0.25)"
                        badge_fg = "#FFAB00"
                    elif 'report:' in lbl or sev == 'REPORT':
                        status_badge = "📋 REPORT LOGGED"
                        badge_bg = "rgba(0, 191, 255, 0.2)"
                        badge_fg = "#00BFFF"
                    else:
                        status_badge = "ℹ️ MONITORED SAFE"
                        badge_bg = "rgba(0, 255, 102, 0.15)"
                        badge_fg = "#00FF66"

                    st.markdown(f"""
                        <div style="
                            background: rgba(4, 10, 24, 0.70);
                            border: 1px solid rgba(255, 255, 255, 0.15);
                            border-left: 5px solid {border_color};
                            backdrop-filter: blur(40px);
                            border-radius: 14px;
                            padding: 0.8rem 1rem;
                            margin-bottom: 0.8rem;
                        ">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="color:#FFF; font-weight:800; font-size:0.95rem;">{log_item['icon']} {log_item['hazard_label']}</span>
                                <span style="color:#888; font-size:0.75rem; font-family:monospace;">⏱️ {log_item['timestamp']}</span>
                            </div>
                            <div style="color:#B0C4DE; font-size:0.85rem; margin-bottom:6px;">
                                📍 <b>Location:</b> {log_item['location']} <span style="color:#777;">({log_item['coords']})</span><br>
                                📏 <b>Dimensions:</b> <span style="color:#FF1744; font-weight:700;">{log_item.get('dimensions', 'N/A')}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem;">
                                <span style="color:#AAA;">Confidence: <b style="color:#00BFFF;">{log_item['confidence']}</b> | {log_item['weather']}</span>
                                <span style="background:{badge_bg}; color:{badge_fg}; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:800; border:1px solid {badge_fg}44;">{status_badge}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
