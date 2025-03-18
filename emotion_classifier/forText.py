from emotion_classifier.utils import load_text_model, preprocess_text
import numpy as np

def classify_text_emotion(text):
    """Classifies the emotion in a text string."""
    tokenizer, model = load_text_model()
    if not tokenizer or not model:
        return "Error: Text model or tokenizer not loaded."

    try:
        padded_sequence = preprocess_text(text, tokenizer)
        predictions = model.predict(padded_sequence)
        class_names = ['happy', 'sad', 'angry', 'neutral', 'surprise']
        predicted_class_index = np.argmax(predictions)
        return class_names[predicted_class_index]
    except Exception as e:
        return f"Error classifying text: {e}"

