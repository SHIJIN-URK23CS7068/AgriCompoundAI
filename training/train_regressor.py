import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset_loader import PlantRegressionDataset
from models.plant_classifier import PlantClassifier
from models.quality_regressor import QualityRegressor
from utils.config import REGRESSOR_PATH, CLASSIFIER_PATH
from utils.logger import logger

def train_regressor(epochs: int = 3, batch_size: int = 4, lr: float = 0.001):
    logger.info("Starting Quality Regressor Training...")

    # 1. Transform setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Dataset load
    try:
        dataset = PlantRegressionDataset(transform=transform)
    except Exception as e:
        logger.error(f"Failed to initialize regression dataset: {e}")
        raise e

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Load trained classifier to extract features
    classifier = PlantClassifier(num_classes=5).to(device)
    if CLASSIFIER_PATH.exists():
        try:
            classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
            logger.info("Loaded trained classifier weights for feature extraction.")
        except Exception as e:
            logger.warning(f"Could not load classifier weights ({e}). Using initialized backbone.")
    else:
        logger.warning("Classifier weights not found. Using initialized backbone for feature extraction.")
    
    classifier.eval() # Freeze backbone

    # Determine image feature dimension
    # Create dummy tensor to pass through extract_features
    dummy = torch.zeros((1, 3, 224, 224)).to(device)
    with torch.no_grad():
        dummy_feat = classifier.extract_features(dummy)
    img_feat_dim = dummy_feat.shape[1]
    logger.info(f"Detected image backbone features dimension: {img_feat_dim}")

    # 5. Initialize Regressor
    model = QualityRegressor(image_feature_dim=img_feat_dim, spectral_dim=23).to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        
        for images, spectral, targets in train_loader:
            images = images.to(device)
            spectral = spectral.to(device)
            targets = targets.to(device)

            # Extract features from frozen classifier
            with torch.no_grad():
                img_features = classifier.extract_features(images)

            optimizer.zero_grad()
            outputs = model(img_features, spectral)
            
            # Loss calculations
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Regression Loss (MSE): {epoch_loss:.4f}")

    # 7. Save model
    torch.save(model.state_dict(), REGRESSOR_PATH)
    logger.info(f"Regressor weights successfully saved to: {REGRESSOR_PATH}")
    return model

if __name__ == "__main__":
    train_regressor()
