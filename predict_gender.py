"""
predict.py

Charge le modèle gender_model.h5 et prédit Homme/Femme pour une image donnée.
Usage : python predict.py chemin/vers/image.jpg
"""

import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
IMG_SIZE = 96
MODEL_PATH = "gender_model.h5"

def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Impossible de trouver le modèle '{MODEL_PATH}'.")
        return

    # Charger le modèle (compile=False pour éviter problèmes d'API)
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print("[ERROR] Impossible de charger le modèle :", e)
        return

    # Lire et prétraiter l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Impossible de lire l'image '{image_path}'.")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prédiction
    pred = model.predict(img)[0]
    homme_pct = pred[0] * 100
    femme_pct = pred[1] * 100

    print(f"Homme  : {homme_pct:.2f}%")
    print(f"Femme  : {femme_pct:.2f}%")

if __name__ == "__main__":
    image_name = input("Image name: ")
    image_path = "data/dataToTest/" + image_name + ".jpg"
    predict_image(image_path)
