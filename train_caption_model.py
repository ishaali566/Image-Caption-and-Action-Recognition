import os
import numpy as np
from models.image_caption_model import ImageCaptionModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def train_caption_model():
    # Paths
    images_path = 'datasets/flickr8k/Images'
    captions_file = 'datasets/flickr8k/captions.txt'

    # Initialize model
    caption_model = ImageCaptionModel(max_length=34, vocab_size=8000)

    # Build feature extractor
    caption_model.build_feature_extractor()

    # Load captions
    print("Loading captions...")
    captions_dict = caption_model.load_captions(captions_file)

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = caption_model.create_tokenizer(captions_dict)

    # Extract features for all images
    print("Extracting image features...")
    features = {}

    image_files = list(captions_dict.keys())[:1000]  # Use subset for faster training

    for img_name in tqdm(image_files):
        img_path = os.path.join(images_path, img_name)
        if os.path.exists(img_path):
            feature = caption_model.extract_features(img_path)
            if feature is not None:
                features[img_name] = feature

    print(f"Extracted features for {len(features)} images")

    # Prepare training data
    print("Preparing training data...")
    X1, X2, y = [], [], []

    for img_name, captions in captions_dict.items():
        if img_name not in features:
            continue

        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=caption_model.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=caption_model.vocab_size)[0]

                X1.append(features[img_name][0])
                X2.append(in_seq)
                y.append(out_seq)

    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    print(f"Training data shape: X1={X1.shape}, X2={X2.shape}, y={y.shape}")

    # Build and train model
    print("Building model...")
    caption_model.build_model()

    print("Training model...")
    caption_model.model.fit(
        [X1, X2], y,
        epochs=20,
        batch_size=64,
        verbose=1
    )

    # Save model
    caption_model.save_model()
    print("Training complete!")


if __name__ == "__main__":
    train_caption_model()