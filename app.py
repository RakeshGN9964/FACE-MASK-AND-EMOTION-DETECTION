import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Face AI", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>😷 Mask + 😊 Emotion Detection (Webcam)</h1>",
    unsafe_allow_html=True
)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    mask = load_model("model/mask_detector.h5")
    emotion = load_model("model/emotion_model.h5")
    return mask, emotion

mask_model, emotion_model = load_models()

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

# ---------------- Video Processor ----------------
class FaceEmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            f1 = cv2.resize(face, (224, 224)) / 255.0
            f1 = f1.reshape(1, 224, 224, 3)

            f2 = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            f2 = f2.reshape(1, 48, 48, 1)

            mask = "Mask" if mask_model.predict(f1, verbose=0)[0][0] > 0.5 else "No Mask"
            emotion = emotions[np.argmax(emotion_model.predict(f2, verbose=0))]

            label = f"{mask} | {emotion}"
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                img, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,0,255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- WebRTC UI ----------------
webrtc_streamer(
    key="face-ai",
    video_processor_factory=FaceEmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
