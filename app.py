import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

# --- Page config ---
st.set_page_config(page_title="🌸 Flower Classifier", layout="wide")

# --- App title ---
st.title("🌸 Flower Image Classifier")

# Display flower types on main page
st.markdown("**This model can predict the following flowers:**")
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
st.write(", ".join([c.capitalize() for c in class_names]))

# --- Sidebar with scrollable instructions ---
instructions = f"""
1. Upload one or multiple flower images (jpg, jpeg, png).  
2. Wait for predictions.  
3. See predicted flower type with probability breakdown.  
4. Download the predictions as CSV.  

**Flower types the model can predict:**  
- {class_names[0].capitalize()}  
- {class_names[1].capitalize()}  
- {class_names[2].capitalize()}  
- {class_names[3].capitalize()}  
- {class_names[4].capitalize()}

You can add more detailed instructions here. The sidebar will scroll if the content is long.
"""

# Wrap instructions in a scrollable div
st.sidebar.markdown(
    f"""
    <div style="height:400px; overflow-y:auto; padding:10px; border:1px solid #ccc;">
        {instructions}
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load model ---
@st.cache_resource
def load_flower_model():
    return load_model("flowers_cnn.keras")

model = load_flower_model()

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Choose flower images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

colors = {
    "daisy": "#FFB6C1",
    "dandelion": "#FFD700",
    "rose": "#FF6347",
    "sunflower": "#FFA500",
    "tulip": "#8A2BE2"
}

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        cols = st.columns([1, 2])
        with cols[0]:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)
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

            result_dict = {"Image": uploaded_file.name, "Predicted": class_names[predicted_class]}
            for i, cname in enumerate(class_names):
                result_dict[cname] = pred_probs[i]
            results.append(result_dict)

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
