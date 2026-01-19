import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(page_title="Face AI", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>😷 Mask + 😊 Emotion Detection</h1>",
    unsafe_allow_html=True
)

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    mask = load_model("model/mask_detector.h5")
    emotion = load_model("model/emotion_model.h5")
    return mask, emotion

mask_model, emotion_model = load_models()

# ---------------- Data ----------------
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
face_cascade = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "📸 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Processing ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected")
    else:
        for (x, y, w, h) in faces:
            # ---- Mask Prediction ----
            face = frame[y:y+h, x:x+w]
            f1 = cv2.resize(face, (224, 224)) / 255.0
            f1 = f1.reshape(1, 224, 224, 3)

            # ---- Emotion Prediction ----
            emo_face = gray[y:y+h, x:x+w]
            f2 = cv2.resize(emo_face, (48, 48)) / 255.0
            f2 = f2.reshape(1, 48, 48, 1)

            mask = "Mask" if mask_model.predict(f1)[0][0] > 0.5 else "No Mask"
            emotion = emotions[np.argmax(emotion_model.predict(f2))]

            label = f"{mask} | {emotion}"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255,0,255), 2
            )

        st.image(frame, caption="Prediction Result", use_column_width=True)
