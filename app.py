# import streamlit as st
# import cv2
# import numpy as np
# from detect import predict_age_gender
# from face_detector import detect_faces
# from collections import deque

# st.title("👤 Age & Gender Detection (Custom DL Model)")

# option = st.radio("Select Input", ["Image Upload", "Webcam"])

# # -------- IMAGE --------
# if option == "Image Upload":
#     uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

#     if uploaded_file:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)

#         faces = detect_faces(img)

#         if len(faces) == 0:
#             st.warning("No face detected ❌")

#         for (x,y,w,h) in faces:

#             pad = 20
#             h_img, w_img = img.shape[:2]

#             x1 = max(0, x - pad)
#             y1 = max(0, y - pad)
#             x2 = min(w_img, x + w + pad)
#             y2 = min(h_img, y + h + pad)

#             face = img[y1:y2, x1:x2]   # ✅ FIXED

#             if face.shape[0] < 50 or face.shape[1] < 50:
#                 continue

#             age, gender = predict_age_gender(face)

#             if age is None:
#                 continue

#             # Draw bounding box
#             cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

#             # Show result cards
#             col1, col2 = st.columns(2)
#             col1.metric("🧑 Gender", gender)
#             col2.metric("🎂 Age", age)

#         st.image(img, channels="BGR")  # ✅ IMPORTANT

# # -------- WEBCAM --------
# else:
#     run = st.checkbox("Start Webcam")

#     FRAME_WINDOW = st.image([])
#     result_placeholder = st.empty()

#     # 🔥 smoothing buffers (ONLY for webcam)
#     age_buffer = deque(maxlen=10)
#     gender_buffer = deque(maxlen=10)

#     cap = cv2.VideoCapture(0)

#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         faces = detect_faces(frame)

#         for (x,y,w,h) in faces:

#             pad = 20
#             h_img, w_img = frame.shape[:2]

#             x1 = max(0, x - pad)
#             y1 = max(0, y - pad)
#             x2 = min(w_img, x + w + pad)
#             y2 = min(h_img, y + h + pad)

#             face = frame[y1:y2, x1:x2]

#             if face.shape[0] < 50 or face.shape[1] < 50:
#                 continue

#             age, gender = predict_age_gender(face)

#             if age is None:
#                 continue

#             # 🔥 smoothing
#             age_buffer.append(age)
#             gender_buffer.append(gender)

#             smooth_age = int(sum(age_buffer) / len(age_buffer))
#             smooth_gender = max(set(gender_buffer), key=gender_buffer.count)

#             # Draw box
#             cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

#             # Show cards
#             with result_placeholder.container():
#                 col1, col2 = st.columns(2)
#                 col1.metric("🧑 Gender", smooth_gender)
#                 col2.metric("🎂 Age", smooth_age)

#         FRAME_WINDOW.image(frame, channels="BGR")

#     cap.release()

import streamlit as st
import cv2
import numpy as np
from detect import predict_age_gender
from face_detector import detect_faces
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque
import av

st.title("👤 Age & Gender Detection (Custom DL Model)")

option = st.radio("Select Input", ["Image Upload", "Camera"])

# ================= IMAGE UPLOAD =================
if option == "Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        faces = detect_faces(img)

        if len(faces) == 0:
            st.warning("No face detected ❌")

        for (x, y, w, h) in faces:

            pad = 20
            h_img, w_img = img.shape[:2]

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)

            face = img[y1:y2, x1:x2]

            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            age, gender = predict_age_gender(face)

            if age is None:
                continue

            # Draw box
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show result cards
            col1, col2 = st.columns(2)
            col1.metric("🧑 Gender", gender)
            col2.metric("🎂 Age", age)

        st.image(img, channels="BGR")


# ================= CAMERA (WEBRTC) =================
else:
    st.write("📷 Live Camera (Browser Based)")

    # 🔥 smoothing buffers
    age_buffer = deque(maxlen=10)
    gender_buffer = deque(maxlen=10)

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            faces = detect_faces(img)

            for (x, y, w, h) in faces:

                pad = 20
                h_img, w_img = img.shape[:2]

                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_img, x + w + pad)
                y2 = min(h_img, y + h + pad)

                face = img[y1:y2, x1:x2]

                if face.shape[0] < 50 or face.shape[1] < 50:
                    continue

                age, gender = predict_age_gender(face)

                if age is None:
                    continue

                # 🔥 smoothing
                age_buffer.append(age)
                gender_buffer.append(gender)

                smooth_age = int(sum(age_buffer) / len(age_buffer))
                smooth_gender = max(set(gender_buffer), key=gender_buffer.count)

                # Draw box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(
                    img,
                    f"{smooth_gender}, {smooth_age}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            return img

    webrtc_streamer(
        key="age-gender",
        video_processor_factory=VideoProcessor
    )