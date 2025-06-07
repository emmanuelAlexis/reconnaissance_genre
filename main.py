import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
import pyttsx3

# --- CONFIGURATION ---
IMG_SIZE = 96
MODEL_PATH = "gender_model.h5"

class GenderClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificateur de Genre (H/F)")
        self.setFixedSize(400, 550)

        # Charger le modèle
        self.model = None
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH, compile=False)
            except Exception as e:
                print("[ERROR] Impossible de charger le modèle :", e)
        else:
            print(f"[ERROR] Modèle '{MODEL_PATH}' introuvable.")

        # Label pour afficher l'image
        self.image_label = QLabel("Aucune image sélectionnée")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(350, 350)

        # Label pour afficher le résultat
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setTextFormat(Qt.RichText)

        # Boutons
        btn_select = QPushButton("📁 Sélectionner une image")
        btn_select.clicked.connect(self.select_image)

        btn_camera = QPushButton("📷 Prendre une photo")
        btn_camera.clicked.connect(self.capture_from_camera)

        # Disposition verticale
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(btn_select)
        layout.addWidget(btn_camera)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

    def select_image(self):
        """
        Ouvre un dialogue pour sélectionner une image depuis le disque.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.display_image(file_path)
            self.predict(file_path)

    def capture_from_camera(self):
        """
        Ouvre la webcam, capture une image, sauve temporairement, puis prédit.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.result_label.setText("❌ Impossible d'accéder à la webcam.")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.result_label.setText("❌ Échec de la capture.")
            return

        # Sauvegarder l'image capturée temporairement
        temp_path = "captured_image.jpg"
        cv2.imwrite(temp_path, frame)

        self.display_image(temp_path)
        self.predict(temp_path)

    def display_image(self, path):
        """
        Affiche l'image (QLabel) à partir du chemin donné.
        """
        image = QImage(path)
        if image.isNull():
            self.image_label.setText("❌ Image invalide.")
            return
        pixmap = QPixmap.fromImage(image).scaled(
            350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def speak(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # Vitesse de la voix
        engine.setProperty("volume", 1.0)  # Volume (0.0 à 1.0)
        engine.setProperty("voice", "french")  # Choisir la voix française
        engine.say(text)
        engine.runAndWait()


    def predict(self, img_path):
        """
        Prédit Homme/Femme sur l'image et met à jour le résultat (QLabel).
        """
        if self.model is None:
            self.result_label.setText("❌ Modèle non chargé.")
            return

        img = cv2.imread(img_path)
        if img is None:
            self.result_label.setText("❌ Impossible de lire l'image.")
            return

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        try:
            pred = self.model.predict(img)[0]
            homme_pct = pred[0] * 100
            femme_pct = pred[1] * 100
            result_html = (
                f"<h3>Résultats :</h3>"
                f"<p style='color:blue;'>👨 Homme : {homme_pct:.2f}%</p>"
                f"<p style='color:deeppink;'>👩 Femme : {femme_pct:.2f}%</p>"
            )
            self.result_label.setText(result_html)

            text_to_speak = f"Genre détecté : Homme: {homme_pct:.2f} pourcent. Et Femme: {femme_pct:.2f} pourcent."
            self.speak(text_to_speak)
        except Exception as e:
            self.result_label.setText(f"❌ Erreur lors de la prédiction : {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GenderClassifierApp()
    window.show()
    sys.exit(app.exec_())

