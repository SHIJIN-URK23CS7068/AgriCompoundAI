import torch
import torch.nn as nn
from utils.logger import logger

class QualityRegressor(nn.Module):
    def __init__(self, image_feature_dim: int = 1280, spectral_dim: int = 23):
        """
        Args:
            image_feature_dim: Dimension of image features (e.g. 768 for ViT, 1280 for MobileNetV2)
            spectral_dim: 18 spectral bands + 5 spectral indices = 23
        """
        super(QualityRegressor, self).__init__()
        
        self.input_dim = image_feature_dim + spectral_dim
        logger.info(f"Initializing Quality Regressor with Input Dim: {self.input_dim} (Image: {image_feature_dim}, Spectral: {spectral_dim})")
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Outputs: [freshness, quality_score, maturity]
        )
        
    def forward(self, img_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_features: [Batch, image_feature_dim]
            spectral_features: [Batch, spectral_dim]
        Returns:
            outputs: [Batch, 3] representing [freshness_pct, quality_score_pct, maturity_val]
        """
        # Concatenate features
        fused = torch.cat([img_features, spectral_features], dim=1)
        logits = self.network(fused)
        
        # Apply bounds using activations
        # Freshness: 0 to 100 %
        # Quality Score: 0 to 100 %
        # Maturity: 0.0 to 1.0
        freshness = torch.sigmoid(logits[:, 0:1]) * 100.0
        quality = torch.sigmoid(logits[:, 1:2]) * 100.0
        maturity = torch.sigmoid(logits[:, 2:3])
        
        return torch.cat([freshness, quality, maturity], dim=1)
