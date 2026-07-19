import streamlit as st
from PIL import Image

from utils.model import load_model
from utils.inference import predict_image

from utils.gradcam_utils import generate_gradcam
from utils.gemini_utils import generate_explanation

# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#f8fafc;
}

div[data-testid="metric-container"]{
    border-radius:12px;
    padding:18px;
    background:white;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

.stAlert{
    border-radius:12px;
}

h1,h2,h3{
    color:#0f172a;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Model (only once)
# ------------------------------

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("🏥 AI Healthcare Assistant")

st.sidebar.markdown("---")

st.sidebar.header("📌 About")

st.sidebar.write(
"""
This app detects **8 skin diseases**
using an EfficientNet-B0 Deep Learning model
trained on the ISIC 2019 Dataset.
"""
)

st.sidebar.markdown("---")

st.sidebar.header("📊 Model")

st.sidebar.write("Model : EfficientNet-B0")
st.sidebar.write("Dataset : ISIC 2019")
st.sidebar.write("Test Accuracy : 75.1%")

st.sidebar.markdown("---")

st.sidebar.header("🛠 Technologies")

st.sidebar.write("""
- PyTorch
- Streamlit
- EfficientNet
- GradCAM
- OpenRouter LLM
""")

st.sidebar.markdown("---")

st.sidebar.success("SOC IIT Bombay Final Project")

# ------------------------------
# Header
# ------------------------------

st.title("🏥 AI Healthcare Assistant")

st.markdown(
"""
### Skin Disease Detection using Deep Learning

Upload a dermoscopic skin image to predict the most likely skin disease.

"""
)

st.divider()

# ------------------------------
# Upload
# ------------------------------

uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing image..."):
        result = predict_image(model, image)

    # ------------------------------
    # Generate GradCAM
    # ------------------------------

    heatmap = generate_gradcam(model, image)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(image, use_container_width=True)

    with col2:

        st.subheader("🔥 AI Attention (Grad-CAM)")

        st.image(heatmap, use_container_width=True)

    st.divider()

    left, right = st.columns([1,1])

    with left:

        st.subheader("🩺 Prediction")

        st.success(result["name"])

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

        st.progress(result["confidence"]/100)

        st.metric(
            "Severity",
            result["severity"]
        )

    with right:

        st.subheader("ℹ Disease Information")

        st.info(result["description"])

        st.warning(result["consult"])

        st.divider()

        st.subheader("Top Predictions")

        for prediction in result["top_predictions"]:
            st.write(f"**{prediction['name']}**")
            st.progress(prediction["confidence"]/100)
            st.caption(f"{prediction['confidence']:.2f}%")

        st.divider()

        st.subheader("🤖 AI Healthcare Explanation")

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(result)

        st.markdown(explanation)

    st.divider()

  
    
    
    