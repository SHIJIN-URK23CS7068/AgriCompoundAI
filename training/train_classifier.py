import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset_loader import PlantClassificationDataset
from models.plant_classifier import PlantClassifier
from utils.config import CLASSIFIER_PATH
from utils.logger import logger

def train_classifier(epochs: int = 3, batch_size: int = 4, lr: float = 0.001):
    logger.info("Starting Plant Classifier Training...")
    
    # 1. Image transformations
    # Since our images are pre-resized to 224x224, standard normalization is sufficient
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load dataset
    try:
        dataset = PlantClassificationDataset(transform=transform)
    except Exception as e:
        logger.error(f"Failed to initialize classification dataset: {e}")
        raise e

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantClassifier(num_classes=5).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100.0 * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    # 5. Save model weights
    torch.save(model.state_dict(), CLASSIFIER_PATH)
    logger.info(f"Classifier weights successfully saved to: {CLASSIFIER_PATH}")
    return model

if __name__ == "__main__":
    train_classifier()
