from emotion_classifier.utils import load_image_model, preprocess_image
import numpy as np

def classify_image_emotion(img_path):
    """Classifies the emotion in an image."""
    model = load_image_model()
    if not model:
        return "Error: Image model not loaded."

    try:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        class_names = ['happy', 'sad', 'negative', 'positive', 'other']
        predicted_class_index = np.argmax(predictions)
        return class_names[predicted_class_index]
    except Exception as e:
        return f"Error classifying image: {e}"
    
    

