import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tensorflow.keras.applications import ResNet50

class TextClassificationModel(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
    
    

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageClassificationModel, self).__init__()
        self.resnet = ResNet50(weights="imagenet", include_top=False)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def create_model(model_type="text", num_classes=2):
    if model_type == "text":
        return TextClassificationModel(num_labels=num_classes)
    elif model_type == "image":
        return ImageClassificationModel(num_classes=num_classes)
