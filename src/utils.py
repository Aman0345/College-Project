import torch
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    """
    Train a model with the given data.
    """
    model.train()
    for epoch in range(num_epochs):
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    """
    Evaluate the trained model.
    """
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())
    
    return accuracy_score(true_labels, predictions)
