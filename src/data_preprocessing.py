import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from PIL import Image
import cv2

def load_text_data(text_file):
    """
    Load text data from a CSV file.
    """
    data = pd.read_csv(text_file)
    return data

def tokenize_text(text_data, tokenizer, max_length=128):
    """
    Tokenize text data using BERT tokenizer.
    """
    tokens = tokenizer(text_data.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return tokens

def preprocess_images(image_folder, target_size=(224, 224)):
    """
    Preprocess images (resize and normalize).
    """
    image_data = []
    for img_file in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).resize(target_size)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        image_data.append(img)
    
    return np.array(image_data)

def preprocess_data(text_file, image_folder, tokenizer):
    """
    Preprocess both text and image data.
    """
    # Text processing
    text_data = load_text_data(text_file)
    tokens = tokenize_text(text_data['text'], tokenizer)
    
    # Image processing
    image_data = preprocess_images(image_folder)
    
    return tokens, image_data
