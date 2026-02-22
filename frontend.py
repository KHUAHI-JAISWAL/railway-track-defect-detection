import streamlit as st
import requests
from PIL import Image

st.title("Railway Track Defect Detection System")

uploaded_file = st.file_uploader(
    "Upload a railway track image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Defect"):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),     # ⭐ VERY IMPORTANT
                        uploaded_file.type
                    )
                }
            )

            if response.status_code == 200:
                result = response.json()
                st.success("Detection Result")
                st.write("Track type :", result["class"])
                st.write("Confidence :", round(result["confidence"] * 100, 2), "%")
            else:
                st.error("Error from server: " + str(response.status_code))
                st.write(response.text)

        except Exception as e:
            st.error(f"Error connecting to server: {e}")