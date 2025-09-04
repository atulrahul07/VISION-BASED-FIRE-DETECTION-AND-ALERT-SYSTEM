import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def classify_image(img_path, model, target_size=(224, 224), threshold=0.75):
    """
    Enhanced classification with proper MobileNetV3 preprocessing
    Returns: (prediction_label, confidence, raw_output)
    """
    # Load image with OpenCV for better error handling
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image from {img_path}")
    
    # Convert color space and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Apply model-specific preprocessing
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    # Get prediction with timing
    predictions = model.predict(img, verbose=0)
    
    # Binary classification interpretation
    fire_prob = predictions[0][0]  # Direct probability value
    confidence = fire_prob if fire_prob > 0.5 else 1 - fire_prob
    label = "Fire" if fire_prob > threshold else "Non-Fire"
    
    return label, confidence, fire_prob

# Load model with custom objects if needed
model = tf.keras.models.load_model("model/fire_detection_model.h5")

# Example usage with detailed output
test_images = [
    "dataset/fire/fire.110.png",
    "dataset/non_fire/non_fire.204.png"
     # Challenging negative example
]

for img_path in test_images:
    try:
        label, confidence, raw = classify_image(img_path, model)
        print(f"""\
🔍 Image: {img_path}
   Prediction: {label}
   Confidence: {confidence:.2%}
   Raw Output: {raw:.4f}
   Threshold: >0.75 required
{'-'*40}""")
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

