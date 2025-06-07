import os
import cv2
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📛 Supprimer les warnings de PIL / JPEG corrompus
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
#
# --- CONFIGURATION ---
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data/UTKFace"
MODEL_SAVE_PATH = "gender_model.h5"

# --- CHARGEMENT DES IMAGES ---
data, labels = [], []
fichiers = os.listdir(DATA_DIR)
print(f"[INFO] Chargement de {len(fichiers)} images depuis {DATA_DIR}")

for i, filename in enumerate(fichiers):
    if i % 500 == 0:
        print(f" - Progression : {i}/{len(fichiers)}")

    try:
        gender = int(filename.split("_")[1])
        if gender not in [0, 1]:
            continue

        img_path = os.path.join(DATA_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(gender)
    except Exception:
        continue

print(f"[INFO] Images valides : {len(data)}")

# --- PRÉPARATION DES DONNÉES ---
X = np.array(data, dtype="float32") / 255.0
y = to_categorical(labels, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"[INFO] Train set : {len(X_train)} - Validation set : {len(X_val)}")

# --- AUGMENTATION DE DONNÉES ---
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# --- CONSTRUCTION DU MODÈLE ---
base_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=base_input)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# --- ENTRAÎNEMENT ---
print("[INFO] Entraînement du modèle...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS
)

# --- SAUVEGARDE ---
model.save(MODEL_SAVE_PATH, include_optimizer=False)
print(f"[INFO] Modèle sauvegardé dans : {MODEL_SAVE_PATH}")
