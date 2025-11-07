import streamlit as st
from PIL import Image
import cv2
import numpy as np

def main():
    st.set_page_config(page_title="Image Viewer", page_icon="📷", layout="wide")
    
    st.title("📷 Image Viewer")
    st.caption("Simple image display and webcam capture")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Webcam (real-time)", "Upload (image)"])

    col1, col2 = st.columns(2)

    with col1:
        if mode == "Upload (image)":
            uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_container_width=True)
        else:
            st.write("Camera Feed")
            camera = st.camera_input("Take a picture")
            if camera:
                img = Image.open(camera)
                st.image(img, caption="Captured Image", use_container_width=True)

    with col2:
        st.write("Instructions:")
        st.write("1. Choose mode: Upload an image or use webcam")
        st.write("2. For best results, ensure good lighting")
        st.write("3. Images will be displayed in real-time")

if __name__ == "__main__":
    main()