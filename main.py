from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os

app = Flask(__name__)

# CORS configure - production ke liye
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Global variables for model
processor = None
model = None

def load_model():
    """Load model on startup"""
    global processor, model
    
    # Agar pehle se loaded hai to skip kar
    if processor is not None and model is not None:
        return True
    
    print("🔄 Loading model from Hugging Face...")
    MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        
        # CPU optimize kar
        model.eval()
        torch.set_num_threads(2)
        
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ⭐ YAHAN PE MODEL LOAD KARO - Gunicorn ke liye
# Yeh module import hote hi run ho jayega
print("=" * 50)
print("🌿 Leaf Disease Detection App")
print("=" * 50)
load_model()

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    global processor, model
    
    # Debug logging add kar
    print(f"📥 Predict request received")
    print(f"Model loaded: {model is not None}")
    print(f"Processor loaded: {processor is not None}")
    print(f"Files in request: {list(request.files.keys())}")
    
    if processor is None or model is None:
        print("❌ Model not loaded!")
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please wait a moment and try again.'
        }), 500
    
    try:
        if 'image' not in request.files:
            print(f"❌ No 'image' key found. Available keys: {list(request.files.keys())}")
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty file'
            }), 400
        
        print(f"📷 Processing image: {image_file.filename}")
        
        # Image process kar
        image = Image.open(image_file.stream).convert('RGB')
        
        # Model ke liye prepare kar
        inputs = processor(images=image, return_tensors="pt")
        
        # Prediction kar
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Top 3 predictions nikal
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k=3)
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            label = model.config.id2label[idx.item()]
            confidence = prob.item() * 100
            predictions.append({
                'disease': label,
                'confidence': f"{confidence:.2f}%",
                'confidence_value': confidence
            })
        
        print(f"✅ Prediction successful: {predictions[0]['disease']}")
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'model_loaded': processor is not None and model is not None
    })

# Local development ke liye (optional - Render pe nahi chalega)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=True,
        host='0.0.0.0',
        port=port,
        threaded=True
    )
