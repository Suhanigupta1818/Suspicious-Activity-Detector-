
import streamlit as st
from transformers import pipeline
import tempfile
import os

# 1. Page Config for Dark/Professional Look
st.set_page_config(page_title="Surveillance AI", page_icon="🛡️", layout="wide")

# Custom CSS to make it look like Gradio
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; }
    .stProgress > div > div > div > div { background-color: #f63366; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("video-classification", model="facebook/timesformer-base-finetuned-k400")

classifier = load_model()

st.title("🛡️ Surveillance AI: Suspicious Activity Detector")
st.markdown("Upload a video to check for suspicious behavior (Fighting, Robbery, etc.)")

# --- EXAMPLE VIDEOS SETUP ---
# Screenshot ke hisaab se paths (Ensure these exist in your /content/ folder)
example_videos = {
    "Example 1 (Fighting)": "/content/dataset/dataset-video-split/test/Fighting047_x264.mp4",
    "Example 2 (Abuse)": "/content/dataset/dataset-video-split/test/Abuse018_x264.mp4",
    "Example 3 (Normal)": "/content/dataset/dataset-video-split/test/Normal_Videos090_x264.mp4"
}

# Session State for examples
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📽️ Video Input")
    uploaded_file = st.file_uploader("Drop Video Here", type=["mp4", "avi"])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        st.video(uploaded_file)

with col2:
    st.subheader("📊 Predictions")
    if st.session_state.video_path:
        if st.button("Submit"):
            with st.spinner('Analyzing...'):
                results = classifier(st.session_state.video_path, top_k=3)
                for res in results:
                    label = res['label'].replace("_", " ").title()
                    st.write(f"**{label}**")
                    st.progress(res['score'])
                    st.write(f"Confidence: {res['score']:.2%}")
    else:
        st.info("Upload a video or select an example below.")

# --- GRADIO-STYLE EXAMPLES SECTION ---
st.markdown("---")
st.subheader("📝 Examples")
cols = st.columns(len(example_videos))

for i, (name, path) in enumerate(example_videos.items()):
    if cols[i].button(name):
        if os.path.exists(path):
            st.session_state.video_path = path
            st.success(f"Selected: {name}. Now click 'Submit' in the Prediction column.")
        else:
            st.error("Example file not found in Colab path!")

st.markdown("---")
st.button("🚩 Flag")
