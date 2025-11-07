import os
import time
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from translator import text_to_speech, translate_text
from utils import extract_landmarks_from_bgr, normalize_landmarks

# No model loaded by default. If you want predictions, place a trained model at
# model/sign_model.h5 and load it into `model` (TensorFlow). For now we provide
# a safe predict stub so the app runs without TensorFlow/MediaPipe.
model = None
labels = None

def predict_from_landmarks(landmarks_flat):
    """Stub prediction: returns (label, prob) if model is loaded, else (None, None)."""
    global model, labels
    if model is None or labels is None:
        return None, None
    try:
        x = normalize_landmarks(landmarks_flat).reshape(1, -1)
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        return labels[idx], float(preds[idx])
    except Exception:
        return None, None

st.set_page_config(page_title="Sign Language Translator", page_icon="🤟", layout="wide")

st.title("🤟 Sign Language Translator")
st.caption("Basic version - Image display and webcam capture")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Webcam (real-time)", "Upload (image)"])
    tts_enabled = st.checkbox("Speak output (gTTS)", value=False)
    translate_enabled = st.checkbox("Translate output", value=False)
    dest_lang = st.text_input("Translate to (ISO code)", value="en")

col1, col2 = st.columns(2)

if mode == "Upload (image)":
    with col1:
        uploaded = st.file_uploader("Upload a hand image (JPG/PNG). Show a clear single-hand gesture.", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            landmarks, annotated = extract_landmarks_from_bgr(frame_bgr)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected hand", width='stretch')
            if landmarks is not None:
                label, prob = predict_from_landmarks(landmarks)
                if label:
                    st.success(f"Prediction: **{label}**  •  Confidence: **{prob:.2%}**")
                    final_text = label
                    if translate_enabled:
                        try:
                            final_text = translate_text(final_text, dest_lang=dest_lang)
                        except Exception as e:
                            st.warning(f"Translation error: {e}")
                    if tts_enabled:
                        try:
                            audio_path = text_to_speech(final_text, lang=dest_lang if translate_enabled else 'en')
                            audio_file = open(audio_path, "rb")
                            st.audio(audio_file.read(), format="audio/mp3")
                            audio_file.close()
                            os.remove(audio_path)
                        except Exception as e:
                            st.warning(f"TTS error: {e}")
                else:
                    st.info("Model not loaded. Train and place model/sign_model.h5.")
            else:
                st.warning("No hand detected. Try a clearer image.")

    with col2:
        st.markdown("### Tips")
        st.write("""
        - Use a plain background and good lighting.
        - Keep one hand in the frame.
        - Start with letters (A, B, C) before words.
        """)

else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not accessible. If running on cloud, use Upload mode or streamlit-webrtc.")
    else:
        frame_slot = st.empty()
        pred_slot = st.empty()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                landmarks, annotated = extract_landmarks_from_bgr(frame)
                frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                if landmarks is not None and model is not None:
                    label, prob = predict_from_landmarks(landmarks)
                    if label:
                        pred_slot.success(f"Prediction: **{label}**  •  Confidence: **{prob:.1%}**")
                        if tts_enabled and prob > 0.7:
                            try:
                                text = label
                                if translate_enabled:
                                    text = translate_text(text, dest_lang=dest_lang)
                                audio_path = text_to_speech(text, lang=dest_lang if translate_enabled else 'en')
                                audio_file = open(audio_path, "rb")
                                st.audio(audio_file.read(), format="audio/mp3")
                                audio_file.close()
                                os.remove(audio_path)
                            except Exception as e:
                                st.warning(f"TTS error: {e}")
                time.sleep(0.03)
        except Exception as e:
            st.info(f"Stopped: {e}")
        finally:
            cap.release()
