import streamlit as st
import requests

st.set_page_config(page_title="PlantDoc AI", layout="wide")

API_URL = "http://127.0.0.1:8000"

# ---------- RESET ----------
def reset_app():
    st.session_state.clear()

# ---------- HEADER ----------
st.title("🌿 PlantDoc Intelligence Console")
st.caption("Diagnose Leaf Disease from Image or Text")

st.button("🔄 Reset", on_click=reset_app)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📸 Image", "📝 Text"])

# ================= IMAGE TAB =================
with tab1:
    col1, col2 = st.columns([2,1])

    # LEFT SIDE
    with col1:
        st.subheader("Leaf Image")

        image_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg"],
            key="image_uploader"
        )

        if image_file:
            st.image(image_file, use_container_width=True)

        if st.button("Predict disease"):
            if image_file:
                response = requests.post(
                    f"{API_URL}/predict-image",
                    files={"file": image_file}
                )
                result = response.json()
                st.session_state["img_result"] = result.get("prediction", "Error")
            else:
                st.warning("Upload an image")

    # RIGHT SIDE
    with col2:
        st.subheader("Prediction")

        if "img_result" in st.session_state:
            st.success(st.session_state["img_result"])
        else:
            st.info("Waiting for prediction...")

        st.subheader("Known Disease Classes")

        # ---------- BOX ----------
        box = st.container(border=True)

        with box:
            known_classes = [
                "Tomato___Bacterial_spot",
                "Tomato___Early_blight",
                "Tomato___Late_blight",
                "Tomato___Leaf_Mold",
                "Tomato___Septoria_leaf_spot",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Grape___Black_rot",
                "Grape___Esca_(Black_Measles)",
                "Pepper_bell___Bacterial_spot"
            ]

            for cls in known_classes:
                st.markdown(
                    f"""
                    <div style="
                        background:#f5f1d5;
                        padding:8px 12px;
                        margin-bottom:6px;
                        border-radius:10px;
                    ">
                    {cls}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ================= TEXT TAB =================
with tab2:
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Enter Symptoms")

        text_input = st.text_area("Describe plant symptoms")

        if st.button("Predict text"):
            if text_input:
                response = requests.post(
                    f"{API_URL}/predict-text",
                    data={"text": text_input}
                )
                result = response.json()
                st.session_state["text_result"] = result.get("prediction", "Error")
            else:
                st.warning("Enter some text")

    with col2:
        st.subheader("Prediction")

        if "text_result" in st.session_state:
            st.success(st.session_state["text_result"])
        else:
            st.info("Waiting for prediction...")
