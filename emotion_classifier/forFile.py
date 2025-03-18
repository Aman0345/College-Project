from emotion_classifier.utils import load_domain_classifier
import os

def analyze_file_emotion(file_path):
    """Analyzes the emotion in a file by domain classification and sentiment analysis."""
    domain_classifier = load_domain_classifier()
    if not os.path.exists(file_path):
        return {"error": "File not found."}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if domain_classifier:
            predicted_domain = domain_classifier.predict([text])[0]
        else:
            predicted_domain = "Unknown"

        sentiment = "Neutral"
        if "happy" in text.lower() or "joy" in text.lower():
            sentiment = "Positive"
        elif "sad" in text.lower() or "depressed" in text.lower():
            sentiment = "Negative"

        return {"domain": predicted_domain, "sentiment": sentiment}
    except Exception as e:
        return {"error": f"Error analyzing file: {e}"}
