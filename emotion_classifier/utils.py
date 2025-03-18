import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib


PROMPT = "You are a sentiment analyzer and you only have the analyze the sentiment of the given input . Give the answer in variour sentiments like positive,negative,happy,sad etc with percentage.You dont have to explain you just have to give the sentiment and a little explaination for your answer ( dont expalin too much 10 to 25 words will be enough)"

def load_image_model(model_path='models/emotion_classifier.h5', img_height=150, img_width=150):
    """Loads the pre-trained image emotion classification model."""
    return tf.keras.models.load_model(model_path)

def load_text_model(tokenizer_path='models/tokenizer.json', model_path='models/text_emotion_model.h5'):
    """Loads the pre-trained text emotion classification model and tokenizer."""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer = Tokenizer.from_json(f.read())
        model = tf.keras.models.load_model(model_path)
        return tokenizer, model
    except Exception as e:
        return None, None

def load_domain_classifier(model_path='models/domain_classifier.pkl'):
    """Loads the pre-trained domain classification model."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        return None

def preprocess_image(img_path, img_height=150, img_width=150):
    """Preprocesses an image for model input."""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def preprocess_text(text, tokenizer, max_length=100):
    """Preprocesses text for model input."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence