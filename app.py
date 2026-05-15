import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model('model/solar_model.h5')

classes = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Solar AI Dashboard",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}

.metric-number {
    font-size: 30px;
    font-weight: bold;
}

.metric-label {
    color: #9CA3AF;
}

</style>
""", unsafe_allow_html=True)

# =========================
# PANEL STATES
# =========================

if "panel_states" not in st.session_state:
    st.session_state.panel_states = {
        i: "#22c55e" for i in range(1, 49)
    }

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Control Panel")

selected_panel = st.sidebar.selectbox(
    "Select Solar Panel",
    [f"Panel {i}" for i in range(1, 49)]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Solar Panel Image",
    type=['jpg', 'jpeg', 'png']
)

run_button = st.sidebar.button("🚀 Run AI Detection")

# =========================
# TITLE
# =========================

st.markdown(
    "<div class='main-title'>☀️ Solar Panel Monitoring Dashboard</div>",
    unsafe_allow_html=True
)

st.write("AI-powered solar panel defect detection using VGG16")

# =========================
# TOP METRICS
# =========================

green_count = list(st.session_state.panel_states.values()).count("#22c55e")
yellow_count = list(st.session_state.panel_states.values()).count("#eab308")
red_count = (
    list(st.session_state.panel_states.values()).count("#ef4444")
    +
    list(st.session_state.panel_states.values()).count("#991b1b")
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>48</div>
        <div class='metric-label'>Total Panels</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{green_count}</div>
        <div class='metric-label'>Healthy Panels</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{yellow_count}</div>
        <div class='metric-label'>Warning Panels</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-number'>{red_count}</div>
        <div class='metric-label'>Critical Panels</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# MAIN LAYOUT
# =========================

left, right = st.columns([2,1])

# =========================
# LEFT SIDE GRID
# =========================

with left:

    st.subheader("🟢 Solar Farm Grid")

    panel_number = 1

    for row in range(6):

        cols = st.columns(8)

        for col in cols:

            color = st.session_state.panel_states[panel_number]

            # Highlight selected panel
            border = "4px solid white" if selected_panel == f"Panel {panel_number}" else "1px solid #333"

            col.markdown(
                f'''
                <div style="
                    width:55px;
                    height:55px;
                    background:{color};
                    border-radius:10px;
                    margin:auto;
                    margin-top:10px;
                    border:{border};
                    box-shadow:0px 0px 15px {color};
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:black;
                    font-weight:bold;
                ">
                    {panel_number}
                </div>
                ''',
                unsafe_allow_html=True
            )

            panel_number += 1

# =========================
# RIGHT SIDE AI SECTION
# =========================

with right:

    st.subheader("🤖 AI Prediction")

    st.write(f"Selected Panel: **{selected_panel}**")

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if run_button:

            # PREPROCESS
            img = image.resize((224,224))
            img_array = np.array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            # PREDICT
            prediction = model.predict(img_array)

            predicted_class = classes[np.argmax(prediction)]

            confidence = np.max(prediction) * 100

            st.success(f"Prediction: {predicted_class}")

            st.progress(int(confidence))

            st.info(f"Confidence: {confidence:.2f}%")

            # PANEL NUMBER
            panel_id = int(selected_panel.split(" ")[1])

            # UPDATE PANEL COLOR
            if predicted_class == 'Clean':

                st.session_state.panel_states[panel_id] = "#22c55e"
                st.markdown("## 🟢 Healthy Panel")

            elif predicted_class == 'Dusty':

                st.session_state.panel_states[panel_id] = "#eab308"
                st.markdown("## 🟡 Dust Detected")

            elif predicted_class == 'Bird-drop':

                st.session_state.panel_states[panel_id] = "#f97316"
                st.markdown("## 🟠 Bird Drop Detected")

            elif predicted_class == 'Snow-Covered':

                st.session_state.panel_states[panel_id] = "#ffffff"
                st.markdown("## ⚪ Snow Covered")

            elif predicted_class == 'Electrical-damage':

                st.session_state.panel_states[panel_id] = "#ef4444"
                st.markdown("## 🔴 Electrical Damage")

            else:

                st.session_state.panel_states[panel_id] = "#991b1b"
                st.markdown("## 🔴 Physical Damage")

            st.subheader("📊 AI Analysis")

            st.write(f"""
            The AI system analyzed the uploaded image and detected
            **{predicted_class}** with a confidence score of
            **{confidence:.2f}%**.

            The selected panel status has been updated in the solar farm grid.
            """)

# =========================
# FOOTER
# =========================

st.write("")
st.write("---")
st.caption("AI-based Solar Panel Monitoring System • VGG16 • Streamlit")