# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

# --- Page config MUST be at the top ---
st.set_page_config(page_title="🌸 Flower Classifier", layout="wide")

# --- Dark Mode CSS ---
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

/* Sidebar background */
.css-1d391kg { 
    background-color: #1A1D23 !important; 
    color: #FFFFFF; 
    padding: 10px;
}

/* Remove collapsible arrow in sidebar instructions */
.css-1d391kg .stMarkdown, .css-1d391kg .stText {
    background-color: transparent !important;
    padding: 0;
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

# --- Load model with caching ---
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

# --- Streamlit UI ---
st.markdown("**This model can predict the following flowers:**")
st.write(", ".join([c.capitalize() for c in class_names]))
st.write("Upload one or more flower images to get predictions with colorful probability bars.")

# --- Sidebar instructions without close arrow ---
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload one or multiple flower images (jpg, jpeg, png).")
st.sidebar.write("2. Wait for predictions.")
st.sidebar.write("3. See predicted flower type with probability breakdown.")
st.sidebar.write("4. Download the predictions as CSV.")
st.sidebar.write("**Flower types the model can predict:**")
for cname in class_names:
    st.sidebar.write(f"- {cname.capitalize()}")

# --- File uploader ---
uploaded_files = st.file_uploader("Choose flower images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.container():
            st.markdown("---")
            cols = st.columns([1, 2])

            # Left column: show image
            with cols[0]:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)

            # Right column: prediction and probabilities
            with cols[1]:
                img_resized = img.resize((150, 150))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Predicting..."):
                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction)
                    pred_probs = prediction[0]

                st.markdown(f"### Predicted: **{class_names[predicted_class]}**")
                st.write("**Class Probabilities:**")
                for i, cname in enumerate(class_names):
                    prob_percent = pred_probs[i] * 100
                    width = max(prob_percent, 2)
                    st.markdown(f"""
                        <div style="margin-bottom:5px;">
                            <strong>{cname}: {prob_percent:.2f}%</strong>
                            <div style="background-color:#e0e0e0; border-radius:10px; width:100%; height:25px;">
                                <div style="background-color:{colors[cname]}; width:{width}%; height:100%; border-radius:10px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Store results
                result_dict = {"Image": uploaded_file.name, "Predicted": class_names[predicted_class]}
                for i, cname in enumerate(class_names):
                    result_dict[cname] = pred_probs[i]
                results.append(result_dict)

    # Download button
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
