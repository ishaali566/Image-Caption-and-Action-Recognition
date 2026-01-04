import tensorflow as tf
from models.action_recognition_model import ActionRecognitionModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os


def train_action_model():
    # Verify dataset path
    data_path = 'datasets/stanford40'

    if not os.path.exists(data_path):
        print(f"Error: Dataset path '{data_path}' not found!")
        print("Please ensure the dataset is in the correct location.")
        return

    print(f"Dataset path: {data_path}")
    print(f"Folders found: {os.listdir(data_path)}")

    # Initialize model
    action_model = ActionRecognitionModel()

    # Build feature extractor
    print("\nBuilding feature extractor...")
    action_model.build_feature_extractor()

    # Load dataset
    print("\nLoading Stanford 40 Actions dataset...")
    try:
        images, labels = action_model.load_stanford40_data(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    action_model.label_encoder = label_encoder

    # Update num_classes
    action_model.num_classes = len(label_encoder.classes_)

    print(f"Number of classes: {action_model.num_classes}")
    print(f"Classes: {list(label_encoder.classes_)[:10]}...")

    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution (samples per class):")
    for cls, count in zip(unique[:5], counts[:5]):
        print(f"  {cls}: {count} images")
    print(f"  ... (showing first 5 classes)")

    # Preprocess images
    print("\nPreprocessing images...")
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images.astype('float32'))

    # Extract features
    print("\nExtracting features from images...")
    features = action_model.feature_extractor.predict(images, batch_size=32, verbose=1)

    print(f"Features shape: {features.shape}")

    # Split data
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Build and train model
    print("\nBuilding classification model...")
    action_model.build_model()

    print("\nModel Summary:")
    action_model.model.summary()

    print("\nTraining model...")
    print("This may take 10-30 minutes depending on your hardware...")

    history = action_model.model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = action_model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Calculate top-k accuracy
    y_pred_prob = action_model.model.predict(X_test, verbose=0)

    # Top-3 accuracy
    top3_pred = np.argsort(y_pred_prob, axis=1)[:, -3:]
    top3_acc = sum([y_test[i] in top3_pred[i] for i in range(len(y_test))]) / len(y_test)
    print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc * 100:.2f}%)")

    # Top-5 accuracy
    top5_pred = np.argsort(y_pred_prob, axis=1)[:, -5:]
    top5_acc = sum([y_test[i] in top5_pred[i] for i in range(len(y_test))]) / len(y_test)
    print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc * 100:.2f}%)")

    # Save model
    print("\nSaving model...")
    action_model.save_model()

    # Save training history
    import json
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }

    os.makedirs('models/saved_models', exist_ok=True)
    with open('models/saved_models/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Model saved to: models/saved_models/action_model.h5")
    print(f"Label encoder saved to: models/saved_models/label_encoder.pkl")
    print("=" * 60)


if __name__ == "__main__":
    train_action_model()


