from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.image_caption_model import ImageCaptionModel
from models.action_recognition_model import ActionRecognitionModel

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get the correct model paths (parent directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

print(f"Base directory: {BASE_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Files in model directory: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")

# Initialize models
print("Loading models...")
caption_model = ImageCaptionModel()
action_model = ActionRecognitionModel()

# Try to load pretrained models with correct paths
caption_model_path = os.path.join(MODEL_DIR, 'caption_model.h5')
caption_tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')

action_model_path = os.path.join(MODEL_DIR, 'action_model.h5')
action_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

print(f"\nLooking for caption model at: {caption_model_path}")
print(f"Caption model exists: {os.path.exists(caption_model_path)}")

print(f"\nLooking for action model at: {action_model_path}")
print(f"Action model exists: {os.path.exists(action_model_path)}")

caption_loaded = caption_model.load_pretrained_model(
    model_path=caption_model_path,
    tokenizer_path=caption_tokenizer_path
)

action_loaded = action_model.load_pretrained_model(
    model_path=action_model_path,
    encoder_path=action_encoder_path
)

if caption_loaded:
    print("✓ Caption model loaded successfully!")
else:
    print("✗ Caption model not found or failed to load")

if action_loaded:
    print("✓ Action model loaded successfully!")
else:
    print("✗ Action model not found or failed to load")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'caption_model_loaded': caption_loaded,
        'action_model_loaded': action_loaded,
        'model_directory': MODEL_DIR
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            response = {}

            # Get image caption
            if caption_loaded:
                try:
                    caption = caption_model.generate_caption(filepath)
                    response['caption'] = caption
                    response['caption_status'] = 'success'
                except Exception as e:
                    response['caption'] = f'Error generating caption: {str(e)}'
                    response['caption_status'] = 'error'
                    print(f"Caption error: {e}")
            else:
                response['caption'] = 'Caption model not trained yet'
                response['caption_status'] = 'model_not_found'

            # Get action recognition
            if action_loaded:
                try:
                    action_result = action_model.predict_action(filepath)
                    response['action'] = action_result
                    response['action_status'] = 'success'
                except Exception as e:
                    response['action'] = {
                        'action': f'Error: {str(e)}',
                        'confidence': 0.0,
                        'all_predictions': []
                    }
                    response['action_status'] = 'error'
                    print(f"Action error: {e}")
            else:
                response['action'] = {
                    'action': 'Model not trained',
                    'confidence': 0.0,
                    'all_predictions': []
                }
                response['action_status'] = 'model_not_found'

            return jsonify(response)

        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass

    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Flask API Server Starting...")
    print("=" * 60)
    print(f"Caption Model: {'✓ Loaded' if caption_loaded else '✗ Not Found'}")
    print(f"Action Model: {'✓ Loaded' if action_loaded else '✗ Not Found'}")
    print("=" * 60)
    print("\nServer running at: http://localhost:5000")
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")

    app.run(debug=True, port=5000)