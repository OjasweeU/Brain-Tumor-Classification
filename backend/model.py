import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model    
from tensorflow.keras.preprocessing.image import img_to_array
import base64
import tensorflow.keras as K

CLASS_DICT = {
    0: 'glioma_tumor',
    1: 'meningioma_tumor',
    2: 'no_tumor',
    3: 'pituitary_tumor'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("WARNING: model.h5 not found. Using MOCK mode for predictions.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_empty_gradcam(image):
    # Just return the grayscale/resized image as base64 as a fallback
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def generate_gradcam(model, img_array, original_img):
    try:
        prediction_idx = np.argmax(model.predict(img_array))
        last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D))
        target_layer = model.get_layer(last_conv_layer.name)

        with tf.GradientTape() as tape:
            gradient_model = tf.keras.models.Model([model.inputs], [target_layer.output, model.output])
            conv2d_out, prediction = gradient_model(img_array)
            loss = prediction[:, prediction_idx]

        gradients = tape.gradient(loss, conv2d_out)
        output = conv2d_out[0]
        weights = tf.reduce_mean(gradients[0], axis=(0, 1))

        activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
        for idx, weight in enumerate(weights):
            activation_map += weight * output[:, :, idx]

        activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
        activation_map = np.maximum(activation_map, 0)
        
        if activation_map.max() - activation_map.min() > 0:
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        
        activation_map = np.uint8(255 * activation_map)
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

        original_img_norm = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
        
        superimposed_img = np.uint8(original_img_norm * 0.5 + heatmap * 0.5)
        
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"GradCAM generation failed: {e}")
        return get_empty_gradcam(original_img)


def predict_image(image_bytes):
    # Convert bytes to numpy array representing the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image file"}

    original_img = image.copy()
    
    # Process image for prediction (similar to training process)
    # We should resize to 240x240 as per efficientnet
    try:
        img_resized = cv2.resize(image, (240, 240))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) # add batch dimension
    except Exception as e:
        return {"error": f"Error during image processing: {e}"}

    if model is None:
        # Mock Response
        import random
        mock_class = random.choice(list(CLASS_DICT.values()))
        mock_gradcam = get_empty_gradcam(cv2.resize(original_img, (240,240)))
        return {
            "prediction": mock_class,
            "confidence": round(random.uniform(0.7, 0.99), 2),
            "gradcam_base64": mock_gradcam,
            "is_mock": True,
            "message": "model.h5 not found! Running in mock mode. Run 'python train.py' to generate the true model."
        }

    # Real Prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    predicted_class = CLASS_DICT.get(class_idx, "Unknown")
    confidence = float(predictions[0][class_idx])

    # GradCAM
    gradcam_b64 = generate_gradcam(model, img_array, cv2.resize(original_img, (240, 240)))

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "gradcam_base64": gradcam_b64,
        "is_mock": False
    }
