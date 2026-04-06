import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import smtplib
import os
from email.message import EmailMessage
from datetime import datetime
import time

# --- CONFIGURATION ---
SENDER_EMAIL = "gsuhani053@gmail.com"
SENDER_PASSWORD = "kuem fujp pnmz oyxe" 
RECEIVER_EMAIL = "gsuhani053@gmail.com" # Maine yahan typo fix kar diya hai (@ missing tha)

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Sentinel | Security", page_icon="🛡️", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3d59; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2564/2564344.png", width=100)
    st.title("AI Sentinel Control")
    st.info("Status: System Online ✅")
    st.divider()
    st.markdown("### System Specs")
    st.write("**Model:** MobileNetV2 + LSTM")
    st.write("**Accuracy:** 80.00%")
    st.write("**Alert Mode:** Email Enabled 📧")

from tensorflow.keras.layers import InputLayer

class PatchedInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        # Jo keywords error de rahe hain, unhe nikal do
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)
        
# --- LOAD MODEL (Modern Native Format) ---
@st.cache_resource
def load_my_model():
    model_path = 'final_improved_model.keras' # Extension change kiya
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' GitHub par nahi mili!")
        return None
    try:
        # Naye format mein koi 'custom_objects' ki zaroorat nahi padti
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Final Loading Error: {e}")
        return None
        # --- MODEL INITIALIZATION ---
model = load_my_model()  # <--- YE LINE MISSING HAI, ISE ADD KARO
     # --- FUNCTIONS ---
def send_email_alert(confidence):
    try:
        current_time = datetime.now().strftime("%I:%M %p | %d %b %Y")
        msg = EmailMessage()
        msg.set_content(f"""
        🚨 SECURITY ALERT TRIGGERED 🚨
        
        System has detected suspicious behavior in the monitored area.
        
        Timestamp: {current_time}
        AI Confidence Score: {confidence:.2f}%
        
        Action Taken: Automated Admin Notification.
        Please review the footage immediately.
        """)
        msg['Subject'] = f"⚠️ CRITICAL ALERT: Suspicious Activity Detected"
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except:
        return False

def extract_frames(video_path, num_frames=15, img_size=(64, 64)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames: return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.array(frames)

# --- MAIN UI ---
st.title("🛡️ Smart Surveillance Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Video Input")
    uploaded_file = st.file_uploader("Upload surveillance footage (MP4, AVI)", type=["mp4", "avi"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.video(uploaded_file)

with col2:
    st.subheader("📊 Analysis Engine")
    if uploaded_file:
        if st.button("START THREAT SCAN"):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            frames = extract_frames("temp_video.mp4")
            
            if frames is not None:
                test_input = np.expand_dims(frames, axis=0).astype('float32') / 255.0
                res = model.predict(test_input)
                prediction = float(np.squeeze(res))
                
                st.write("### Detection Result:")
                if prediction > 0.5:
                    conf = prediction * 100
                    st.error(f"⚠️ **SUSPICIOUS ACTIVITY DETECTED**")
                    st.metric(label="Threat Confidence", value=f"{conf:.2f}%", delta="CRITICAL", delta_color="inverse")
                    
                    with st.status("Initiating Protocol...", expanded=True) as status:
                        st.write("Logging activity...")
                        st.write("Contacting server...")
                        if send_email_alert(conf):
                            st.write("Email Notification Sent! 📧")
                        else:
                            st.write("Email Service Offline. ❌")
                        status.update(label="Alert Protocol Complete", state="complete", expanded=False)
                else:
                    conf = (1 - prediction) * 100
                    st.success(f"✅ **NORMAL BEHAVIOR**")
                    st.metric(label="Safety Confidence", value=f"{conf:.2f}%", delta="SAFE")
            else:
                st.warning("Video too short for AI analysis.")
    else:
        st.info("Waiting for video upload to begin scanning...")

st.divider()
st.caption("Developed by Suhani | AI & ML Project 2026")
