import torch
import torch.nn as nn
from utils.logger import logger

class ShelfLifeLSTM(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, num_layers: int = 1, max_days: float = 15.0):
        """
        Args:
            input_dim: Number of features per timestep (e.g. Temp, Humidity, Freshness, Quality, NDVI, WBI)
            hidden_dim: LSTM hidden state size
            num_layers: Number of LSTM layers
            max_days: Maximum shelf life prediction limit
        """
        super(ShelfLifeLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_days = max_days
        
        logger.info(f"Initializing Shelf Life LSTM with Input Dim: {input_dim}, Hidden Dim: {hidden_dim}")
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Input_Dim]
        Returns:
            shelf_life: [Batch, 1] representing remaining shelf life in days
        """
        # x is [Batch, Seq_Len, Input_Dim]
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last timestep
        last_out = out[:, -1, :] # [Batch, Hidden_Dim]
        logits = self.fc(last_out)
        
        # Shelf life should be non-negative and bounded by max_days
        shelf_life = torch.sigmoid(logits) * self.max_days
        return shelf_life
