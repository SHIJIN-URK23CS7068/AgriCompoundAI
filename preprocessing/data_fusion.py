import numpy as np
import torch
from utils.logger import logger

class DataFuser:
    def __init__(self):
        pass

    def fuse_features(self, image_features: dict, spectral_features: dict) -> np.ndarray:
        """
        Performs early-stage fusion by concatenating statistical color features 
        with spectral bands and spectral indices.
        Returns:
            np.ndarray: A 1D array of fused features.
        """
        # Extract color features list
        color_vec = np.array([
            image_features.get("mean_R", 0.0) / 255.0,
            image_features.get("mean_G", 0.0) / 255.0,
            image_features.get("mean_B", 0.0) / 255.0,
            image_features.get("mean_H", 0.0) / 180.0,
            image_features.get("mean_S", 0.0) / 255.0,
            image_features.get("mean_V", 0.0) / 255.0,
            image_features.get("mean_L", 0.0) / 100.0,
            (image_features.get("mean_a_lab", 0.0) + 128.0) / 255.0,
            (image_features.get("mean_b_lab", 0.0) + 128.0) / 255.0,
        ], dtype=np.float32)

        # Extract spectral bands (18 values)
        spectral_vec = spectral_features["vector"] # shape (18,)

        # Extract spectral indices
        indices_dict = spectral_features["indices"]
        indices_vec = np.array([
            indices_dict.get("NDVI", 0.0),
            indices_dict.get("WBI", 0.0) / 5.0, # scaled
            indices_dict.get("CI", 0.0) / 10.0, # scaled
            indices_dict.get("CurcuminIndex", 0.0),
            indices_dict.get("AnthocyaninIndex", 0.0) / 20.0 # scaled
        ], dtype=np.float32)

        # Concatenate everything: 9 + 18 + 5 = 32 dimensions
        fused = np.concatenate([color_vec, spectral_vec, indices_vec], axis=0)
        logger.debug(f"Fused features vector generated. Shape: {fused.shape}")
        return fused

    def fuse_dl_features(self, backbone_tensor: torch.Tensor, spectral_tensor: torch.Tensor) -> torch.Tensor:
        """
        Fuses high-dimensional deep learning image features with raw spectral bands.
        Args:
            backbone_tensor: [Batch, Feature_Dim] (e.g. [1, 512] or [1, 768])
            spectral_tensor: [Batch, 18]
        Returns:
            torch.Tensor: Fused representation
        """
        # Concatenate along feature dimension
        fused_tensor = torch.cat([backbone_tensor, spectral_tensor], dim=1)
        logger.debug(f"Fused deep learning features shape: {fused_tensor.shape}")
        return fused_tensor
