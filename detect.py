import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gdown

MODEL_PATH = "models/age_gender_model.keras"

# 🔥 Auto download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    
    print("📥 Downloading model...")

    url = "https://drive.google.com/uc?id=1lYC0Q9SmlYHwJKjb6NeG1HVZO_HaFUrF"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

def predict_age_gender(face_img):
    try:
        # 🔥 Convert BGR → RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize
        face_img = cv2.resize(face_img, (128,128))

        # Normalize
        face_img = face_img.astype("float32") / 255.0

        # Expand dims
        face_img = np.expand_dims(face_img, axis=0)

        # Predict
        pred = model.predict(face_img, verbose=0)

        gender_prob = pred[0][0][0]
        age_pred = pred[1][0][0]

        # Gender mapping (UTKFace: 0=Male, 1=Female)
        gender = "Female" if gender_prob > 0.5 else "Male"

        # Clamp age
        age = int(max(0, min(100, age_pred)))

        return age, gender

    except:
        return None, None