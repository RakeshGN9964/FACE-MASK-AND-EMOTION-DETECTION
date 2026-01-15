
import streamlit as st
import cv2
import numpy as np

from tensorflow.keras.models import load_model

st.set_page_config(page_title="Face AI", layout="wide")
st.markdown("<h1 style='text-align:center;'>😷 Mask + 😊 Emotion Detection</h1>", unsafe_allow_html=True)

mask_model = load_model("model/mask_detector.h5")
emotion_model = load_model("model/emotion_model.h5")

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

run = st.toggle("🎥 Start Webcam")

frame_slot = st.empty()
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        f1 = cv2.resize(face,(224,224))/255.0
        f1 = f1.reshape(1,224,224,3)

        f2 = cv2.resize(gray[y:y+h,x:x+w],(48,48))/255.0
        f2 = f2.reshape(1,48,48,1)

        mask = "Mask" if mask_model.predict(f1)[0][0] > 0.5 else "No Mask"
        emotion = emotions[np.argmax(emotion_model.predict(f2))]

        label = f"{mask} | {emotion}"
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,255),2)

    frame_slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

cap.release()
