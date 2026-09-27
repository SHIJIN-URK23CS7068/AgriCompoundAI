import sys
from pathlib import Path

# Add project root to path to ensure modules load correctly
sys.path.append(str(Path(__file__).resolve().parent))

from dataset.generate_mock_dataset import create_mock_dataset
from training.train_classifier import train_classifier
from training.train_regressor import train_regressor
from training.train_lstm import train_lstm
from utils.logger import logger

def main():
    logger.info("=== Starting AgriCompoundAI Master Training Pipeline ===")
    
    try:
        # Step 1: Generate dataset
        logger.info("Step 1: Generating Mock Dataset...")
        create_mock_dataset()
        
        # Step 2: Train Classifier
        logger.info("Step 2: Training Plant Classifier (ViT / MobileNetV2)...")
        train_classifier(epochs=3)
        
        # Step 3: Train Regressor
        logger.info("Step 3: Training Quality Regressor Head...")
        train_regressor(epochs=3)
        
        # Step 4: Train LSTM
        logger.info("Step 4: Training Shelf Life LSTM...")
        train_lstm(epochs=5)
        
        logger.info("=== Training Pipeline Completed Successfully! ===")
        logger.info("All model weights have been saved in: models/weights/")
        
    except Exception as e:
        logger.error(f"Critical error in training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
