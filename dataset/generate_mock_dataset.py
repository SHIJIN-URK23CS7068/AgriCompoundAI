import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
from utils.config import DATASET_DIR, BASE_DIR
from utils.logger import logger
from multispectral.sensor_reader import MultispectralSensorReader

def create_mock_dataset():
    logger.info("Initializing Mock Dataset Generation...")
    
    classes = ["Basil", "Lavender", "Spinach", "Tomato", "Turmeric"]
    
    # 1. Setup paths
    classifier_dir = DATASET_DIR / "classifier"
    regression_dir = DATASET_DIR / "regression"
    lstm_dir = DATASET_DIR / "lstm"
    
    os.makedirs(classifier_dir, exist_ok=True)
    os.makedirs(regression_dir, exist_ok=True)
    os.makedirs(lstm_dir, exist_ok=True)
    
    for cls in classes:
        os.makedirs(classifier_dir / cls, exist_ok=True)
        
    sensor_reader = MultispectralSensorReader()
    regression_rows = []
    
    # 2. Generate augmented images and regression labels
    static_img_dir = BASE_DIR / "web" / "static" / "images"
    
    for cls in classes:
        src_path = static_img_dir / f"{cls.lower()}.png"
        
        # Load source image or create fallback
        if src_path.exists():
            base_img = cv2.imread(str(src_path))
        else:
            logger.warning(f"Source image {src_path} not found. Creating fallback canvas.")
            base_img = np.zeros((300, 300, 3), dtype=np.uint8)
            base_img.fill(50)
            cv2.circle(base_img, (150, 150), 90, (100, 200, 100), -1)
            cv2.putText(base_img, cls, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        base_img = cv2.resize(base_img, (224, 224))
        
        # Generate 15 augmented images per class
        for i in range(15):
            # Augmentation options
            # A. Random rotation
            angle = random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
            aug_img = cv2.warpAffine(base_img, M, (224, 224))
            
            # B. Random brightness and contrast
            alpha = random.uniform(0.8, 1.2) # Contrast
            beta = random.randint(-20, 20)   # Brightness
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
            
            # C. Random Gaussian noise
            noise = np.random.normal(0, 5, aug_img.shape).astype(np.int16)
            aug_img = np.clip(aug_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save image
            img_rel_path = f"classifier/{cls}/{cls.lower()}_aug_{i}.png"
            img_abs_path = DATASET_DIR / img_rel_path
            cv2.imwrite(str(img_abs_path), aug_img)
            
            # Create regression target variables
            freshness = round(random.uniform(40.0, 100.0), 2)
            # Quality score is highly correlated with freshness
            quality = round(freshness * random.uniform(0.85, 1.0), 2)
            maturity = round(random.uniform(0.3, 0.95), 2)
            
            # Get simulated spectral vector (18 bands)
            spec_dict = sensor_reader._read_simulated(cls, freshness)
            # Format spectral vector as comma-separated string
            spec_vector_str = ",".join([str(spec_dict[wl]) for wl in spec_dict])
            
            regression_rows.append({
                "image_path": img_rel_path,
                "spectral_vector": spec_vector_str,
                "freshness": freshness,
                "quality": quality,
                "maturity": maturity
            })
            
    # Save regression csv
    df_reg = pd.DataFrame(regression_rows)
    df_reg.to_csv(regression_dir / "labels.csv", index=False)
    logger.info(f"Generated regression/labels.csv and saved {len(regression_rows)} augmented images.")
    
    # 3. Generate LSTM sequence data
    # We will generate 30 sequence objects (history tracking over 5 days)
    lstm_rows = []
    
    for seq_id in range(30):
        # Pick a random plant class
        cls = random.choice(classes)
        # Starting conditions
        freshness = random.uniform(90.0, 98.0)
        quality = freshness * random.uniform(0.9, 0.98)
        remaining_life = random.uniform(9.0, 14.0)
        
        # Temp/Humidity
        base_temp = random.uniform(20.0, 28.0)
        base_hum = random.uniform(45.0, 65.0)
        
        # Generate 5 timesteps (days) of decay
        for step in range(5):
            # Degradation rates
            deg_rate = random.uniform(4.0, 10.0) # Decay per day
            freshness = max(20.0, freshness - deg_rate * (1.0 + (base_temp - 22.0) * 0.05)) # higher temp decays faster
            quality = max(15.0, freshness * random.uniform(0.85, 0.95))
            remaining_life = max(0.0, remaining_life - 1.0 - random.uniform(-0.2, 0.5))
            
            # Environmental conditions
            temp = base_temp + random.uniform(-1.0, 1.0)
            humidity = base_hum + random.uniform(-3.0, 3.0)
            
            # Generate simulated spectral features to calculate NDVI/WBI
            spec_dict = sensor_reader._read_simulated(cls, freshness)
            ndvi = (spec_dict[760] - spec_dict[680]) / max(0.001, (spec_dict[760] + spec_dict[680]))
            wbi = spec_dict[900] / max(0.001, spec_dict[940])
            
            lstm_rows.append({
                "sequence_id": f"seq_{seq_id:03d}",
                "step": step,
                "temp": round(temp, 2),
                "humidity": round(humidity, 2),
                "freshness": round(freshness, 2),
                "quality": round(quality, 2),
                "NDVI": round(ndvi, 4),
                "WBI": round(wbi, 4),
                "remaining_shelf_life": round(remaining_life, 2)
            })
            
    df_lstm = pd.DataFrame(lstm_rows)
    df_lstm.to_csv(lstm_dir / "sequences.csv", index=False)
    logger.info(f"Generated lstm/sequences.csv with {len(df_lstm)} sequence steps.")
    logger.info("Mock dataset successfully generated!")

if __name__ == "__main__":
    create_mock_dataset()
