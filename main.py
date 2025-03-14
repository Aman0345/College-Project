import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from src.data_preprocessing import preprocess_data
from src.model import create_model
from src.utils import train_model, evaluate_model

def main():
    # Load data and preprocess
    text_file = "data/text/data.csv"  # Your text data
    image_folder = "data/images/"  # Your image data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens, images = preprocess_data(text_file, image_folder, tokenizer)

    # Create data loaders
    train_dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], images)
    train_loader = DataLoader(train_dataset, batch_size=16)

    # Define model and optimizer
    model = create_model(model_type="text", num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_loader, optimizer, criterion, num_epochs=3)

    # Evaluate the model
    accuracy = evaluate_model(model, train_loader)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
