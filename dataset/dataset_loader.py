import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from utils.logger import logger
from utils.config import DATASET_DIR

class PlantClassificationDataset(Dataset):
    def __init__(self, transform=None):
        self.classifier_dir = DATASET_DIR / "classifier"
        self.transform = transform
        self.classes = sorted(["Basil", "Lavender", "Spinach", "Tomato", "Turmeric"])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        if not self.classifier_dir.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Classification dataset directory not found at: {self.classifier_dir}\n"
                f"Please run the mock dataset generator first: python dataset/generate_mock_dataset.py"
            )

        # Gather samples
        for class_name in self.classes:
            class_path = self.classifier_dir / class_name
            if not class_path.exists():
                raise FileNotFoundError(
                    f"\n[ERROR] Class directory '{class_name}' is missing under: {self.classifier_dir}\n"
                    f"Please run dataset/generate_mock_dataset.py to generate mock dataset files."
                )
            
            img_files = list(class_path.glob("*.png")) + list(class_path.glob("*.jpg"))
            for img_file in img_files:
                self.samples.append((img_file, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise ValueError(
                f"\n[ERROR] No image files found in class subdirectories under: {self.classifier_dir}\n"
                f"Please place training images under the respective folders or run: python dataset/generate_mock_dataset.py"
            )

        logger.info(f"Loaded Classification Dataset with {len(self.samples)} samples across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at {img_path}: {e}")

        if self.transform:
            img = self.transform(img)
        else:
            # Basic tensor fallback if no transform is provided
            img = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0

        return img, label


class PlantRegressionDataset(Dataset):
    def __init__(self, transform=None):
        self.regression_dir = DATASET_DIR / "regression"
        self.csv_path = self.regression_dir / "labels.csv"
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Regression labels CSV file not found at: {self.csv_path}\n"
                f"Please run: python dataset/generate_mock_dataset.py"
            )

        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise IOError(f"Failed to read CSV at {self.csv_path}: {e}")

        required_cols = ["image_path", "spectral_vector", "freshness", "quality", "maturity"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"\n[ERROR] Regression CSV is missing required column: '{col}'\n"
                    f"File path: {self.csv_path}"
                )

        logger.info(f"Loaded Regression Dataset with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_str = row["image_path"]
        # Resolve path relative to dataset directory or absolute
        img_path = Path(img_path_str)
        if not img_path.exists():
            # Try relative path resolve
            img_path = DATASET_DIR / img_path_str
            if not img_path.exists():
                raise FileNotFoundError(f"Image file listed in regression CSV not found: {img_path_str}")

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at {img_path}: {e}")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float() / 255.0

        # Load spectral vector (comma-separated string in CSV)
        try:
            spec_list = [float(x) for x in row["spectral_vector"].split(",")]
            # Combine 18 spectral bands + 5 mock indices = 23 features
            # (We will compute indices on the fly or load them from the CSV if present,
            # for robustness, we compute them here or append placeholders to total 23)
            # Let's say the spectral vector in CSV has 18 elements.
            # We'll compute the 5 spectral indices on the fly to build the 23-element vector!
            spec_vector_18 = np.array(spec_list, dtype=np.float32)
            
            # Simple index computations
            ndvi = (spec_vector_18[12] - spec_vector_18[10]) / max(0.001, (spec_vector_18[12] + spec_vector_18[10])) # 730 vs 680
            wbi = spec_vector_18[14] / max(0.001, spec_vector_18[16]) # 810 vs 900
            ci = (spec_vector_18[12] / max(0.001, spec_vector_18[10])) - 1
            curcumin_idx = (spec_vector_18[7] - spec_vector_18[2]) / max(0.001, (spec_vector_18[7] + spec_vector_18[2])) # 585 vs 460
            antho_idx = spec_vector_18[12] / max(0.001, spec_vector_18[5]) # 730 vs 535
            
            indices = np.array([ndvi, wbi / 5.0, ci / 10.0, curcumin_idx, antho_idx / 20.0], dtype=np.float32)
            spec_features = np.concatenate([spec_vector_18, indices])
            spec_tensor = torch.from_numpy(spec_features).float()
        except Exception as e:
            raise ValueError(f"Failed to parse spectral vector in row {idx}: {e}")

        # Targets
        targets = torch.tensor([
            float(row["freshness"]),
            float(row["quality"]),
            float(row["maturity"])
        ], dtype=torch.float32)

        return img_tensor, spec_tensor, targets


class PlantShelfLifeDataset(Dataset):
    def __init__(self, seq_len: int = 5):
        self.lstm_dir = DATASET_DIR / "lstm"
        self.csv_path = self.lstm_dir / "sequences.csv"
        self.seq_len = seq_len

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] LSTM sequence CSV file not found at: {self.csv_path}\n"
                f"Please run: python dataset/generate_mock_dataset.py"
            )

        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise IOError(f"Failed to read CSV at {self.csv_path}: {e}")

        # Columns expected: sequence_id, step, temp, humidity, freshness, quality, NDVI, WBI, remaining_shelf_life
        required_cols = ["sequence_id", "step", "temp", "humidity", "freshness", "quality", "NDVI", "WBI", "remaining_shelf_life"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"\n[ERROR] LSTM CSV is missing required column: '{col}'\n"
                    f"File path: {self.csv_path}"
                )

        # Group by sequence_id
        self.grouped = self.df.groupby("sequence_id")
        self.sequence_ids = list(self.grouped.groups.keys())
        
        logger.info(f"Loaded LSTM Dataset with {len(self.sequence_ids)} sequences of length {seq_len}.")

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        group = self.grouped.get_group(seq_id).sort_values("step")

        if len(group) < self.seq_len:
            # Pad or duplicate
            features_df = group[["temp", "humidity", "freshness", "quality", "NDVI", "WBI"]]
            features = features_df.values.astype(np.float32)
            # Pad by repeating the first element
            padding = np.repeat(features[0:1], self.seq_len - len(group), axis=0)
            features = np.concatenate([padding, features], axis=0)
        else:
            # Take last seq_len records
            features_df = group.tail(self.seq_len)[["temp", "humidity", "freshness", "quality", "NDVI", "WBI"]]
            features = features_df.values.astype(np.float32)

        # Scale features for LSTM stability
        # temp (scale by /35.0), humidity (/100.0), freshness (/100.0), quality (/100.0)
        features[:, 0] = features[:, 0] / 35.0
        features[:, 1] = features[:, 1] / 100.0
        features[:, 2] = features[:, 2] / 100.0
        features[:, 3] = features[:, 3] / 100.0
        # NDVI and WBI are already bounded roughly

        features_tensor = torch.from_numpy(features).float()
        
        # Target shelf life of the last step in sequence
        target_val = float(group.iloc[-1]["remaining_shelf_life"])
        target = torch.tensor([target_val], dtype=torch.float32)

        return features_tensor, target
