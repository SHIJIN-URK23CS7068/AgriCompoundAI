import torch
import torch.nn as nn
import torchvision.models as models
from utils.logger import logger

class PlantClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(PlantClassifier, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = "vit_b_16"
        
        # Try loading Vision Transformer (ViT-B/16)
        try:
            logger.info("Attempting to load ViT-B/16 backbone...")
            # Use weights=ViT_B_16_Weights.DEFAULT or weights='DEFAULT'
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            # ViT-B/16 has a 'heads' attribute where heads.head is the classifier
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, num_classes)
            logger.info("Successfully loaded ViT-B/16 backbone with custom head.")
        except Exception as e:
            logger.warning(f"Failed to load ViT-B/16 backbone ({e}). Falling back to MobileNetV2.")
            self.backbone_name = "mobilenet_v2"
            
            try:
                logger.info("Attempting to load MobileNetV2 with default weights...")
                self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
                logger.info("Successfully loaded MobileNetV2 with custom head.")
            except Exception as e_inner:
                logger.warning(f"Could not load MobileNetV2 with pretrained weights ({e_inner}). Initializing MobileNetV2 from scratch (no download required).")
                # Initialize without weights (no-internet safety)
                self.backbone = models.mobilenet_v2(weights=None)
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
                logger.info("Initialized unweighted MobileNetV2.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, 3, 224, 224] image tensor
        Returns:
            logits: [Batch, num_classes]
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature vectors prior to classification head (useful for quality regression).
        """
        if self.backbone_name == "vit_b_16":
            # For ViT, pass through encoder and extract class token
            # PyTorch ViT has a _process_input method, or we can use the forward with hook
            # A simple way to get features for ViT:
            # We can run the backbone's components manually or register a hook.
            # Alternatively, we can just run the classifier and get logits, but if we want features:
            # In torchvision ViT, the forward path is:
            # x = self._process_input(x)
            # x = self.class_token + x
            # x = self.encoder(x)
            # x = x[:, 0]  # Class token
            # Let's write a clean extraction based on torchvision structure
            b = self.backbone
            x = b._process_input(x)
            n = x.shape[0]
            # Expand the class token to the batch size
            batch_class_token = b.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = b.encoder(x)
            # Classifier "token" is the first token
            features = x[:, 0]
            return features
        else:
            # For MobileNetV2, features are output by the features block followed by global pool
            features = self.backbone.features(x)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            return features
