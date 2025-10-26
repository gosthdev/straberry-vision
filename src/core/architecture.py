"""
Arquitectura del modelo SGSNet
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SGSNet(nn.Module):
    """
    Strawberry Grading System Network
    Red neuronal para detección de estados de madurez de fresas
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 3

        # Backbone: MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.backbone = mobilenet.features

        # Feature Pyramid Network (FPN) - Multi-escala
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(576, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Detection heads mejorados
        self.detection_head = self._make_detection_head(128)

        self._initialize_weights()

    def _make_detection_head(self, in_channels):
        """Crea la cabeza de detección con múltiples capas convolucionales"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, self.num_anchors * (5 + self.num_classes), 1)
        )

    def _initialize_weights(self):
        """Inicializa los pesos de las capas convolucionales y batch norm"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass del modelo
        
        Args:
            x: Tensor de entrada [B, 3, H, W]
            
        Returns:
            predictions: Tensor de predicciones [B, num_anchors*(5+num_classes), H', W']
        """
        features = self.backbone(x)
        fpn_out1 = self.fpn_conv1(features)
        fpn_out2 = self.fpn_conv2(fpn_out1)
        predictions = self.detection_head(fpn_out2)
        return predictions
