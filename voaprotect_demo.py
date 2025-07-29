import os
import time
import json
import queue
import qrcode
import random
import tempfile
import sounddevice as sd
import streamlit as st
import vosk
from PIL import Image
from io import BytesIO
from gtts import gTTS
import numpy as np
import geocoder
import folium
from streamlit_folium import st_folium
from datetime import datetime
from streamlit.components.v1 import html

# --- Configuration dicts ---
INSTRUCTIONS = {
    "English": {
        "record_button": "Start Voice Recording",
        "symptoms_prompt": "Say: fever, chills, headache, vomiting, fatigue, nausea, muscle pain, diarrhea, sore throat, eye pain, dizziness, or confusion.",
        "example_prompt": "For example: 'I have a fever and fatigue'",
        "start_instruction": "Recording will start in a few seconds. You will have 10 seconds to say one or more symptoms clearly.",
        "recognized_text": "You said:",
        "matched_symptoms": "Matched symptoms:",
        "triage_risk": "Triage Risk:",
        "outbreak_risk": "Outbreak Risk:",
        "recommendation": "Recommendation:",
        "low": "🟢 Low",
        "moderate": "🟠 Moderate",
        "high": "🔴 High",
        "summary_intro": "### 📍 Report Summary:",
        "triage_explain": "**Triage Risk** evaluates **your symptoms** and their severity.",
        "outbreak_explain": "**Outbreak Risk** checks **your area** for signs of an outbreak.",
        "temperature_label": "Temperature:",
        "humidity_label": "Humidity:",
        "risk_map_title": "Risk Location Map",
        "download_qr": "Download QR Code",
        "reminder_sent": "QR code ready for health worker scan.",
        "feedback_prompt": "Any additional notes or feedback?",
        "submit_feedback": "Submit Feedback",
        "feedback_saved": "Thank you! Your feedback has been saved.",
        "new_recording": "New Recording",
        "view_log": "View Log"
    },
    "French": {
        "record_button": "Commencer l'enregistrement vocal",
        "symptoms_prompt": "Dites : fièvre, frissons, mal de tête, vomissements, fatigue, nausée, douleurs musculaires, diarrhée, maux de gorge, douleur oculaire, vertiges ou confusion.",
        "example_prompt": "Par exemple : 'J'ai de la fièvre et des frissons'",
        "start_instruction": "L'enregistrement commencera dans quelques secondes. Vous aurez 10 secondes pour dire un ou plusieurs symptômes.",
        "recognized_text": "Vous avez dit :",
        "matched_symptoms": "Symptômes reconnus :",
        "triage_risk": "Risque de triage :",
        "outbreak_risk": "Risque d’épidémie :",
        "recommendation": "Recommandation :",
        "low": "🟢 Faible",
        "moderate": "🟠 Modéré",
        "high": "🔴 Élevé",
        "summary_intro": "### 📍 Résumé du rapport :",
        "triage_explain": "**Le risque de triage** évalue **vos symptômes** et leur gravité.",
        "outbreak_explain": "**Le risque d’épidémie** analyse **votre région** pour détecter une éventuelle épidémie.",
        "temperature_label": "Température :",
        "humidity_label": "Humidité :",
        "risk_map_title": "Carte des risques",
        "download_qr": "Télécharger le code QR",
        "reminder_sent": "Code QR prêt pour le travailleur de santé.",
        "feedback_prompt": "Vos commentaires ou remarques ?",
        "submit_feedback": "Envoyer le retour",
        "feedback_saved": "Merci ! Votre retour a été enregistré.",
        "new_recording": "Nouvel enregistrement",
        "view_log": "Afficher le journal"
    }
}

SYMPTOMS = {
    "English": [
        "fever", "chills", "headache", "vomiting", "fatigue", "nausea",
        "muscle pain", "diarrhea", "sore throat", "eye pain", "dizziness", "confusion"
    ],
    "French": [
        "fièvre", "frissons", "mal de tête", "vomissements", "fatigue", "nausée",
        "douleurs musculaires", "diarrhée", "maux de gorge", "douleur oculaire", "vertiges", "confusion"
    ]
}

# Page configuration
st.set_page_config(page_title="VoaProtect Demo", layout="wide")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    language = st.selectbox("🌐 Language", ["English", "French"])
    st.write("---")
    if st.button(INSTRUCTIONS[language]["record_button"], key="btn_start"):
        st.session_state.start = True
        st.session_state.done  = False
        st.session_state.results = []
        st.session_state.matched = []

# Title
instr = INSTRUCTIONS[language]
st.title("VoaProtect - Malaria Voice Triage AI")

# Text-to-speech helper
def speak(text):
    tts = gTTS(text=text, lang="en" if language=="English" else "fr")
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
        tts.save(fp.name)
        os.system(f"afplay '{fp.name}'")

# Geolocation retrieval
def get_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return f"{g.city}, {g.country}", g.latlng
    except:
        pass
    return "Not available", (None, None)

location, (lat, lon) = get_location()
humidity, temperature = "75%", "28°C"

# Initialize session state once
for key, default in {
    'start': False,
    'done': False,
    'results': [],
    'matched': [],
    'triage': None,
    'outbreak': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Recording block with spinner
def record_and_process():
    # Show and speak symptom prompts
    st.info(instr['symptoms_prompt'])
    st.markdown(instr['example_prompt'])
    speak(instr['symptoms_prompt']); time.sleep(1)
    speak(instr['example_prompt']);    time.sleep(1)

    # Show start instruction before recording
    st.info(instr['start_instruction'])
    speak(instr['start_instruction']); time.sleep(0.5)

    # Spinner for the actual recording period
    with st.spinner("🎙️ Recording for 10 seconds…"):
        # Vosk setup
        model_dir = 'model' if language=='English' else 'model-fr'
        if not os.path.isdir(model_dir):
            st.error(f"Model folder not found: '{model_dir}'")
            return
        model = vosk.Model(model_dir)
        rec   = vosk.KaldiRecognizer(model, 16000)
        q     = queue.Queue()

        # Capture audio
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype='int16',
            channels=1,
            callback=lambda d, f, t, s: q.put(bytes(d))
        ):
            t0 = time.time()
            while time.time() - t0 < 10:
                data = q.get()
                if rec.AcceptWaveform(data):
                    txt = json.loads(rec.Result()).get('text', '')
                    if txt:
                        st.session_state.results.append(txt)
            final = json.loads(rec.FinalResult()).get('text','')
            if final:
                st.session_state.results.append(final)

    # Post-process
    full = ' '.join(st.session_state.results)
    st.session_state.matched = [s for s in SYMPTOMS[language] if s.lower() in full.lower()]
    cnt = len(st.session_state.matched)
    st.session_state.triage   = instr['high'] if cnt>=6 else instr['moderate'] if cnt>=3 else instr['low']
    st.session_state.outbreak = random.choice([instr['low'], instr['moderate'], instr['high']])
    st.session_state.done = True

# Trigger recording logic once
if st.session_state.start and not st.session_state.done:
    record_and_process()

# Display results in tabs
if st.session_state.done:
    tab1, tab2 = st.tabs(["Results", "Logs"])

    # Results tab
    with tab1:
        full = ' '.join(st.session_state.results)
        st.markdown(f"**{instr['recognized_text']}** {full}")
        mstr = ', '.join(st.session_state.matched) or 'None'
        st.markdown(f"**{instr['matched_symptoms']}** {mstr}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Triage Risk", st.session_state.triage)
            st.progress(1 if st.session_state.triage==instr['low'] else 2 if st.session_state.triage==instr['moderate'] else 3)
        with col2:
            st.metric("Outbreak Risk", st.session_state.outbreak)
            st.progress(1 if st.session_state.outbreak==instr['low'] else 2 if st.session_state.outbreak==instr['moderate'] else 3)

        st.markdown(instr['triage_explain'])
        st.markdown(instr['outbreak_explain'])

        payload = {
            'location': location,
            'language': language,
            'symptoms': st.session_state.matched,
            'triage': st.session_state.triage,
            'outbreak': st.session_state.outbreak
        }
        img = qrcode.make(json.dumps(payload, ensure_ascii=False))
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        st.image(buf, caption=instr['reminder_sent'])
        st.download_button(
            instr['download_qr'], data=buf,
            file_name='voaprotect.png', mime='image/png', key='btn_dl'
        )

        if lat and lon:
            st.subheader(instr['risk_map_title'])
            m = folium.Map(location=[lat, lon], zoom_start=12)
            folium.Marker(
                [lat, lon], popup=f"{st.session_state.triage}, {st.session_state.outbreak}"
            ).add_to(m)
            st_folium(m, width=700)

    # Logs tab
    with tab2:
        fb = st.text_area(instr['feedback_prompt'], key='ta_fb')
        if st.button(instr['submit_feedback'], key='btn_fb'):
            entry = {
                'timestamp': datetime.now().isoformat(),
                'language': language,
                'location': location,
                'symptoms': st.session_state.matched,
                'triage': st.session_state.triage,
                'outbreak': st.session_state.outbreak,
                'notes': fb
            }
            path = os.path.join(tempfile.gettempdir(), 'voaprotect_logs.json')
            logs = json.load(open(path)) if os.path.exists(path) else []
            logs.append(entry)
            json.dump(logs, open(path, 'w'), indent=2)
            st.success(instr['feedback_saved'])

        if st.button(instr['view_log'], key='btn_view'):
            path = os.path.join(tempfile.gettempdir(), 'voaprotect_logs.json')
            if os.path.exists(path):
                st.json(json.load(open(path)))
            else:
                st.info("No reports logged yet.")
