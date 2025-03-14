from transformers import pipeline

def sentiment_analysis(text):
    """
    Perform sentiment analysis using a pretrained BERT model.
    """
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)
    return result
