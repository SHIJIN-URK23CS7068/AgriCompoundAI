import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset_loader import PlantShelfLifeDataset
from models.shelf_life_lstm import ShelfLifeLSTM
from utils.config import LSTM_PATH
from utils.logger import logger

def train_lstm(epochs: int = 5, batch_size: int = 4, lr: float = 0.005):
    logger.info("Starting Shelf Life LSTM Training...")

    # 1. Dataset load
    try:
        dataset = PlantShelfLifeDataset(seq_len=5)
    except Exception as e:
        logger.error(f"Failed to initialize LSTM dataset: {e}")
        raise e

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Model setup
    # Input dimension: 6 features (Temp, Humidity, Freshness, Quality, NDVI, WBI)
    model = ShelfLifeLSTM(input_dim=6, hidden_dim=32, num_layers=1, max_days=15.0).to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - LSTM Loss (MSE): {epoch_loss:.4f}")

    # 5. Save model
    torch.save(model.state_dict(), LSTM_PATH)
    logger.info(f"LSTM weights successfully saved to: {LSTM_PATH}")
    return model

if __name__ == "__main__":
    train_lstm()
