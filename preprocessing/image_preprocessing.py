import cv2
import numpy as np
import torch
from utils.config import IMAGE_SIZE
from utils.logger import logger

class ImagePreprocessor:
    def __init__(self):
        self.target_size = IMAGE_SIZE
        # ImageNet standardization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_for_dl(self, img_bgr: np.ndarray) -> torch.Tensor:
        """
        Resizes, standardizes, and converts BGR image to PyTorch tensor format [C, H, W]
        """
        # Resize
        resized = cv2.resize(img_bgr, self.target_size)
        # BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = resized_rgb.astype(np.float32) / 255.0
        # Standardize
        standardized = (normalized - self.mean) / self.std
        # Transpose to [C, H, W]
        transposed = np.transpose(standardized, (2, 0, 1))
        # Convert to torch tensor
        tensor = torch.from_numpy(transposed).float()
        return tensor

    def extract_color_features(self, img_bgr: np.ndarray) -> dict:
        """
        Extracts average values across RGB, HSV, and Lab color spaces
        """
        # BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # BGR to HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # BGR to Lab
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

        mean_rgb = np.mean(img_rgb, axis=(0, 1))
        mean_hsv = np.mean(img_hsv, axis=(0, 1))
        mean_lab = np.mean(img_lab, axis=(0, 1))

        color_features = {
            "mean_R": float(mean_rgb[0]),
            "mean_G": float(mean_rgb[1]),
            "mean_B": float(mean_rgb[2]),
            "mean_H": float(mean_hsv[0]),
            "mean_S": float(mean_hsv[1]),
            "mean_V": float(mean_hsv[2]),
            "mean_L": float(mean_lab[0]),
            "mean_a_lab": float(mean_lab[1]),
            "mean_b_lab": float(mean_lab[2])
        }
        
        logger.debug(f"Extracted color features: {color_features}")
        return color_features
