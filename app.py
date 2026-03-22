
import streamlit as st
from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="Surveillance AI", page_icon="🛡️")
@st.cache_resource
def load_model():
    return pipeline("video-classification", model="facebook/timesformer-base-finetuned-k400")

classifier = load_model()
st.title("🛡️ Suspicious Activity Detection")
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    st.video(uploaded_file)
    if st.button("Analyze Activity"):
        with st.spinner('Analyzing...'):
            results = classifier(tfile.name, top_k=3)
            for res in results:
                st.write(f"**{res['label']}**: {res['score']:.2%}")
                st.progress(res['score'])
