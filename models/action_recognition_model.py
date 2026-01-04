import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle


class ActionRecognitionModel:
    def __init__(self, num_classes=40, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.label_encoder = None
        self.feature_extractor = None

    def build_feature_extractor(self):
        """Build CNN feature extractor using MobileNetV2"""
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=(*self.img_size, 3))
        base_model.trainable = False

        self.feature_extractor = Sequential([
            base_model,
            GlobalAveragePooling2D()
        ])

        print("Feature extractor built")

    def build_model(self):
        """Build complete action recognition model"""
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(1280,)),  # MobileNetV2 output
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Action recognition model built")
        return self.model

    def extract_features(self, image_path):
        """Extract features from image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

            features = self.feature_extractor.predict(img, verbose=0)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def load_stanford40_data(self, data_path):
        """Load Stanford 40 Actions dataset"""
        images = []
        labels = []

        print("Loading Stanford 40 Actions dataset...")

        # Get all action folders
        action_folders = [f for f in os.listdir(data_path)
                          if os.path.isdir(os.path.join(data_path, f))]

        print(f"Found {len(action_folders)} action classes")

        for action_folder in tqdm(action_folders, desc="Loading actions"):
            action_path = os.path.join(data_path, action_folder)

            # Get action name (remove underscores for better display)
            action_name = action_folder.replace('_', ' ')

            # Load all images from this action folder
            image_files = [f for f in os.listdir(action_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in image_files:
                img_path = os.path.join(action_path, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)

                    images.append(img)
                    labels.append(action_name)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

        print(f"Loaded {len(images)} images from {len(set(labels))} action classes")

        if len(images) == 0:
            raise ValueError("No images loaded! Check your dataset path.")

        return np.array(images), np.array(labels)

    def predict_action(self, image_path):
        """Predict action from image"""
        if self.model is None or self.label_encoder is None:
            raise ValueError("Model not loaded")

        features = self.extract_features(image_path)
        if features is None:
            return {"action": "unknown", "confidence": 0.0}

        prediction = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))

        action_name = self.label_encoder.inverse_transform([predicted_class])[0]

        # Get top 5 predictions
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        all_predictions = [
            {
                "action": self.label_encoder.inverse_transform([i])[0],
                "confidence": float(prediction[0][i])
            }
            for i in top_indices
        ]

        return {
            "action": action_name,
            "confidence": confidence,
            "all_predictions": all_predictions
        }

    def save_model(self, model_path='models/saved_models/action_model.h5',
                   encoder_path='models/saved_models/label_encoder.pkl'):
        """Save model and label encoder"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)

        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")

    def load_pretrained_model(self, model_path='models/saved_models/action_model.h5',
                              encoder_path='models/saved_models/label_encoder.pkl'):
        """Load pretrained model"""
        if not os.path.exists(model_path):
            print("No pretrained model found")
            return False

        self.build_feature_extractor()
        self.model = load_model(model_path)

        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Update num_classes based on loaded encoder
        self.num_classes = len(self.label_encoder.classes_)

        print("Action model loaded successfully")
        print(f"Model recognizes {self.num_classes} action classes")
        return True