import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

import src.config as config
from src.grad_cam import GradCAM

# Page Configuration
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor MRI Diagnostic Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .prediction-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model(model_path):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def get_grad_cam_engine(_model):
    if _model is None:
        return None
    return GradCAM(_model)

def main():
    st.markdown('<p class="main-header">🧠 NeuroScan AI — Clinical MRI Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Brain Tumor Classification with Grad-CAM Explainable AI (XAI)</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🩺 Clinical Diagnostics Dashboard")
    st.sidebar.markdown("---")
    
    model = load_cached_model(config.MODEL_SAVE_PATH)
    
    if model is None:
        st.error(f"⚠️ Model checkpoint not found at `{config.MODEL_SAVE_PATH}`.")
        st.info("Please train the model first by running `python main.py --train` from your terminal or CLI!")
        return

    st.sidebar.success("✅ Model Loaded: **EfficientNet Transfer Architecture**")
    
    # Input Selection: Upload or Pick Sample from Test Dataset
    input_mode = st.sidebar.radio("Input Source:", ["📂 Pick Sample from Test Dataset", "⬆️ Upload Custom MRI Scan"])
    
    image_to_analyze = None
    selected_label = None

    if input_mode == "📂 Pick Sample from Test Dataset":
        sample_class = st.sidebar.selectbox("Select Tumor Category:", config.CLASS_NAMES)
        class_dir = os.path.join(config.TEST_DIR, sample_class)
        if os.path.exists(class_dir):
            sample_files = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            selected_file = st.sidebar.selectbox("Select MRI Scan:", sample_files)
            if selected_file:
                img_path = os.path.join(class_dir, selected_file)
                image_to_analyze = Image.open(img_path).convert('RGB')
                selected_label = sample_class
        else:
            st.sidebar.warning(f"No test images found in {class_dir}")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload Brain MRI (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image_to_analyze = Image.open(uploaded_file).convert('RGB')
            selected_label = "Custom Upload"

    # Alpha Slider for Grad-CAM overlay
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔥 XAI Settings")
    alpha_val = st.sidebar.slider("Grad-CAM Overlay Opacity:", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

    if image_to_analyze is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("📸 Input Brain MRI Scan")
            st.image(image_to_analyze, caption=f"Source: {selected_label}", use_container_width=True)

        with col2:
            st.subheader("🔍 Real-Time Diagnostic Analysis")
            with st.spinner("Executing neural inference & computing spatial Grad-CAM activation..."):
                # Preprocess image
                img_resized = image_to_analyze.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
                
                # Grad-CAM Engine
                grad_cam = get_grad_cam_engine(model)
                heatmap, pred_idx, probs = grad_cam.compute_heatmap(img_array)
                
                pred_class = config.CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100

                # Create Colormap Overlay
                heatmap_resized = cv2.resize(heatmap, (config.IMG_SIZE[0], config.IMG_SIZE[1]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                superimposed_float = img_array.astype(np.float32) * (1.0 - alpha_val) + heatmap_colored.astype(np.float32) * alpha_val
                superimposed_img = np.clip(superimposed_float, 0, 255).astype(np.uint8)

            # Display prediction banner
            badge_color = "#10B981" if pred_class != "no_tumor" else "#3B82F6"
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {badge_color};">
                <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: #64748B;">Predicted Pathology</span>
                <div class="prediction-title" style="color: {badge_color};">{pred_class.upper().replace('_', ' ')}</div>
                <span style="font-size: 1.2rem; font-weight: 600; color: #334155;">Diagnostic Confidence: <b>{confidence:.2f}%</b></span>
            </div>
            """, unsafe_allow_html=True)

            # Display Grad-CAM Superimposed Image
            st.image(superimposed_img, caption=f"Grad-CAM Tumor Localization Heatmap (Pathology Focus)", use_container_width=True)

        # Full Probability Breakdown
        st.markdown("---")
        st.subheader("📊 Multi-Class Softmax Probability Distribution")
        
        prob_df = pd.DataFrame({
            "Tumor Category": [cls.replace('_', ' ').title() for cls in config.CLASS_NAMES],
            "Probability (%)": [float(p * 100) for p in probs]
        }).sort_values(by="Probability (%)", ascending=False)

        st.bar_chart(prob_df.set_index("Tumor Category"), height=250)
        
        # Clinical Interpretation Notes
        st.markdown("### 🧬 Clinical Interpretation")
        if pred_class == "glioma_tumor":
            st.info("**Glioma Tumor:** Gliomas originate in the glial cells of the brain or spinal cord. They are common primary brain tumors and can range from low-grade benign to high-grade aggressive glioblastoma.")
        elif pred_class == "meningioma_tumor":
            st.info("**Meningioma Tumor:** Meningiomas arise from the meninges (the membranes surrounding the brain and spinal cord). The vast majority are non-cancerous (benign) and grow slowly over time.")
        elif pred_class == "pituitary_tumor":
            st.info("**Pituitary Tumor:** Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas, but they can disrupt hormonal balance or press against optic nerves.")
        else:
            st.success("**No Tumor Detected:** The MRI scan does not exhibit structural abnormalities consistent with Glioma, Meningioma, or Pituitary tumor mass.")

if __name__ == "__main__":
    main()
