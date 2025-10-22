# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

# --- Page config ---
st.set_page_config(page_title="🌸 Flower Classifier", layout="wide")

# --- Dark Mode & Scrollable Sidebar CSS ---
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1A1D23 !important;
        color: #FFFFFF;
        padding: 10px;
        height: calc(100vh - 2rem);
        overflow-y: auto;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button {
        background-color: #2C2F38;
        color: white;
        border-radius: 8px;
    }

    /* Markdown headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }

    /* Probability bars text */
    div strong {
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# --- App title ---
st.title("🌸 Flower Image Classifier")

# --- Load model ---
@st.cache_resource
def load_flower_model():
    return load_model("flowers_cnn.keras")

model = load_flower_model()

# --- Class names and colors ---
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
colors = {
    "daisy": "#FFB6C1",
    "dandelion": "#FFD700",
    "rose": "#FF6347",
    "sunflower": "#FFA500",
    "tulip": "#8A2BE2"
}

# --- Sidebar instructions (scrollable) ---
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload one or multiple flower images (jpg, jpeg, png).  
2. Wait for predictions.  
3. See predicted flower type with probability breakdown.  
4. Download the predictions as CSV.  

**Flower types the model can predict:**  
- Daisy  
- Dandelion  
- Rose  
- Sunflower  
- Tulip  

You can add more instructions here if needed. The sidebar will scroll automatically if the content is too long.
""")

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Choose flower images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

results = []  # Store results

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.container():
            st.markdown("---")
            cols = st.columns([1, 2])

            # Left column: display image
            with cols[0]:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)

            # Right column: prediction
            with cols[1]:
                # Preprocess image
                img_resized = img.resize((150, 150))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                with st.spinner("Predicting..."):
                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction)
                    pred_probs = prediction[0]

                # Display predicted class
                st.markdown(f"### Predicted: **{class_names[predicted_class]}**")

                # Display probability bars
                st.write("**Class Probabilities:**")
                for i, cname in enumerate(class_names):
                    prob_percent = pred_probs[i] * 100
                    width = max(prob_percent, 2)
                    st.markdown(
                        f"""
                        <div style="margin-bottom:5px;">
                            <strong>{cname}: {prob_percent:.2f}%</strong>
                            <div style="background-color:#e0e0e0; border-radius:10px; width:100%; height:25px;">
                                <div style="background-color:{colors[cname]}; width:{width}%; height:100%; border-radius:10px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Store results
                result_dict = {"Image": uploaded_file.name, "Predicted": class_names[predicted_class]}
                for i, cname in enumerate(class_names):
                    result_dict[cname] = pred_probs[i]
                results.append(result_dict)

    # Download CSV
    if results:
        df = pd.DataFrame(results)
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_buffer,
            file_name="flower_predictions.csv",
            mime="text/csv"
        )
