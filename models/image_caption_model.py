import numpy as np
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle


class ImageCaptionModel:
    def __init__(self, max_length=34, vocab_size=8000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.model = None
        self.feature_extractor = None

    def build_feature_extractor(self):
        """Build InceptionV3 feature extractor"""
        inception = InceptionV3(weights='imagenet')
        # Remove the last layer
        self.feature_extractor = Model(inception.input, inception.layers[-2].output)
        print("Feature extractor loaded")

    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            img = load_img(image_path, target_size=(299, 299))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.inception_v3.preprocess_input(img)

            feature = self.feature_extractor.predict(img, verbose=0)
            return feature
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def load_captions(self, captions_file):
        """Load captions from file"""
        captions_dict = {}

        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                image_name = parts[0]
                caption = parts[1]

                # Clean caption
                caption = caption.lower()
                caption = 'startseq ' + caption + ' endseq'

                if image_name not in captions_dict:
                    captions_dict[image_name] = []
                captions_dict[image_name].append(caption)

        print(f"Loaded {len(captions_dict)} images with captions")
        return captions_dict

    def create_tokenizer(self, captions_dict):
        """Create and fit tokenizer"""
        all_captions = []
        for captions in captions_dict.values():
            all_captions.extend(captions)

        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<unk>')
        self.tokenizer.fit_on_texts(all_captions)

        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        return self.tokenizer

    def build_model(self, embedding_dim=256, units=512):
        """Build the caption generation model (CNN + LSTM) - Using Concatenation"""

        # Image feature input
        image_input = Input(shape=(2048,), name='image_input')
        image_dense = Dropout(0.5)(image_input)
        image_dense = Dense(embedding_dim, activation='relu', name='image_embedding')(image_dense)

        # Caption sequence input
        caption_input = Input(shape=(self.max_length,), name='caption_input')
        caption_embed = Embedding(self.vocab_size, embedding_dim, mask_zero=True, name='caption_embedding')(
            caption_input)
        caption_dropout = Dropout(0.5)(caption_embed)
        caption_lstm = LSTM(units, name='caption_lstm')(caption_dropout)

        # Merge both inputs using concatenation
        decoder = Concatenate(name='merge')([image_dense, caption_lstm])
        decoder = Dense(units, activation='relu', name='decoder_dense1')(decoder)
        decoder = Dropout(0.5)(decoder)
        decoder = Dense(units // 2, activation='relu', name='decoder_dense2')(decoder)
        output = Dense(self.vocab_size, activation='softmax', name='output')(decoder)

        # Create model
        self.model = Model(inputs=[image_input, caption_input], outputs=output)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        print("Model built successfully")
        print(f"\nModel Architecture:")
        self.model.summary()
        return self.model

    def generate_caption(self, image_path):
        """Generate caption for an image"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load or train the model first.")

        # Extract features
        feature = self.extract_features(image_path)
        if feature is None:
            return "Error: Could not process image"

        # Generate caption word by word
        caption = 'startseq'

        for _ in range(self.max_length):
            # Encode caption
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)

            # Predict next word
            prediction = self.model.predict([feature, sequence], verbose=0)
            predicted_id = np.argmax(prediction)

            # Get word from index
            word = None
            for w, idx in self.tokenizer.word_index.items():
                if idx == predicted_id:
                    word = w
                    break

            if word is None or word == 'endseq':
                break

            caption += ' ' + word

        # Remove startseq and clean
        caption = caption.replace('startseq', '').strip()
        return caption

    def save_model(self, model_path='models/saved_models/caption_model.h5',
                   tokenizer_path='models/saved_models/tokenizer.pkl'):
        """Save model and tokenizer"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save(model_path)

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        print(f"Model saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")

    def load_pretrained_model(self, model_path='models/saved_models/caption_model.h5',
                              tokenizer_path='models/saved_models/tokenizer.pkl'):
        """Load pretrained model and tokenizer"""
        if not os.path.exists(model_path):
            print("No pretrained model found. Using InceptionV3 for basic captions.")
            self.build_feature_extractor()
            return False

        self.model = load_model(model_path)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.build_feature_extractor()
        print("Model and tokenizer loaded successfully")
        return True