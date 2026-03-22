import streamlit as st
from transformers import pipeline
import tempfile
import os

# 1. Page Configuration (Exact title and icon like Gradio screenshot)
st.set_page_config(
    page_title="Surveillance AI: Suspicious Activity Detector", 
    page_icon="🛡️",
    layout="wide" # Structured layout ke liye wide mode use kar rahe hain
)

# 2. Caching Model Loading (Loads only once, saving time)
@st.cache_resource
def load_model():
    st.info("🔄 Loading Deep Learning Model (TimeSformer)... Please wait.")
    return pipeline("video-classification", model="facebook/timesformer-base-finetuned-k400")

# Model instance
classifier = load_model()

# 3. Main Title & Description (Gradio-like style)
st.title("🛡️ Surveillance AI: Suspicious Activity Detector")
st.markdown("---")
st.markdown("#### Upload a video to check for suspicious behavior (Fighting, Robbery, etc.)")
st.markdown("---")

# 4. Input & Output Layout (Exact Gradio-like structure)
# Hum 2 columns banayenge: Left mein Input Video, Right mein Results
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📽️ Input Video")
    uploaded_file = st.file_uploader("Drop Video Here or Click to Upload", type=["mp4", "avi"])
    
    if uploaded_file is not None:
        # Save the uploaded video temporarily for the pipeline
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        
        # Display the video with playback controls
        st.video(uploaded_file)
    else:
        # Visual placeholder like Gradio's empty state
        st.info("Waiting for a video file to be uploaded...")

with col2:
    st.subheader("📊 Model Predictions")
    
    if uploaded_file is not None and st.button("🔍 Detect Suspicious Activity"):
        with st.spinner('🎬 Analyzing temporal dynamics...'):
            try:
                # Run inference on the temporary file
                results = classifier(tfile_path, top_k=3)
                
                st.success("✅ Analysis Complete!")
                
                # Gradio-like structured output using columns and progress bars
                for res in results:
                    # Formatting the label (e.g., 'punching' -> 'Punching')
                    readable_label = res['label'].replace("_", " ").title()
                    score = res['score']
                    
                    st.markdown(f"**{readable_label}**")
                    st.progress(score) # Professional progress bar
                    st.markdown(f"{score:.2%} Confidence")
                    st.markdown("---") # Line separator for each prediction
                    
            except Exception as e:
                st.error(f"Error during video analysis: {e}")
            
            # Clean up temporary file
            finally:
                os.unlink(tfile_path)
    else:
        # Gradio-like placeholder for output area
        st.write("Results will appear here once you upload a video and click 'Detect'.")

# 5. Examples & Technical Details Expander (Gradio-like extras)
st.markdown("---")
with st.expander("📝 Implementation Details & Capabilities", expanded=False):
    st.markdown("""
    - **Architecture:** Transformers-based video classification (TimeSformer).
    - **Features:** Temporal-Spatial attention mechanisms for motion tracking.
    - **Baseline:** Pre-trained on Kinetics-400 human action dataset.
    - **Capabilities:** Detection of standard interactions such as Fighting, Shaking Hands, Falling, etc.
    """)

# 6. Flagging Placeholder (Gradio-like UI elements)
st.markdown("---")
st.button("🚩 Flag as False Alarm (Placeholder)")
