# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

# --- Load model with caching ---
@st.cache_resource
def load_flower_model():
    return load_model("flowers_cnn.keras")

model = load_flower_model()

# --- Class names and colors ---
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
colors = {
    "daisy": "#FFB6C1",       # Light Pink
    "dandelion": "#FFD700",   # Gold
    "rose": "#FF6347",         # Tomato Red
    "sunflower": "#FFA500",   # Orange
    "tulip": "#8A2BE2"         # BlueViolet
}

# --- Streamlit UI ---
st.set_page_config(page_title="🌸 Flower Classifier", layout="wide")
st.title("🌸 Flower Image Classifier")
st.write("Upload one or more flower images to get predictions with colorful probability bars.")

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload one or multiple flower images (jpg, jpeg, png).  
2. Wait for predictions.  
3. See predicted flower type with probability breakdown.  
4. Download the predictions as CSV.
""")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose flower images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

results = []  # Store results for download

if uploaded_files:
    # Create cards for each image
    for uploaded_file in uploaded_files:
        with st.container():
            st.markdown("---")  # separator
            cols = st.columns([1, 2])
            
            # Left column: show image
            with cols[0]:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Right column: prediction and probabilities
            with cols[1]:
                # Preprocess
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
                
                # Display probabilities with visible colored bars
                st.write("**Class Probabilities:**")
                for i, cname in enumerate(class_names):
                    prob_percent = pred_probs[i] * 100
                    width = max(prob_percent, 2)  # ensure minimum visibility
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

    # Download button for predictions
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
