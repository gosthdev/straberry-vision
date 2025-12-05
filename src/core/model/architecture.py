"""
Arquitectura del modelo SGSNet v2 - Multi-escala
Mejoras:
- Backbone MobileNetV3-Large con extracción multi-escala
- FPN verdadero con conexiones laterales y top-down pathway
- 3 cabezas de detección (P3, P4, P5) para multi-escala
- 9 anchors totales (3 por escala) con aspect ratios variados
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Mejora la atención espacial y de canales
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x


class FPN(nn.Module):
    """
    Feature Pyramid Network con conexiones laterales verdaderas
    Genera 3 niveles de features: P3 (64x64), P4 (32x32), P5 (16x16)
    """
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: Lista de canales de entrada [C3, C4, C5]
            out_channels: Canales de salida para todas las escalas
        """
        super().__init__()
        
        # Conexiones laterales (1x1 conv para reducir canales)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        
        # Smooth layers (3x3 conv para refinar features después de fusion)
        self.smooth_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Attention modules para cada escala
        self.attention_p5 = CBAM(out_channels)
        self.attention_p4 = CBAM(out_channels)
        self.attention_p3 = CBAM(out_channels)
        
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, c3, c4, c5):
        """
        Top-down pathway con conexiones laterales
        Forzamos FP32 para evitar overflow en FP16
        
        Args:
            c3: Features de stride 8 (alto res)
            c4: Features de stride 16 (medio res)
            c5: Features de stride 32 (bajo res)
            
        Returns:
            p3, p4, p5: Features piramidales
        """
        # Convertir a float32 para estabilidad numérica
        c3 = c3.float()
        c4 = c4.float()
        c5 = c5.float()
        
        # Top level (P5)
        p5 = self.lateral_c5(c5)
        p5 = self.smooth_p5(p5)
        p5 = self.attention_p5(p5)
        
        # P4 = upsample(P5) + lateral(C4)
        p4 = F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4 = p4 + self.lateral_c4(c4)
        p4 = self.smooth_p4(p4)
        p4 = self.attention_p4(p4)
        
        # P3 = upsample(P4) + lateral(C3)
        p3 = F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p3 = p3 + self.lateral_c3(c3)
        p3 = self.smooth_p3(p3)
        p3 = self.attention_p3(p3)
        
        return p3, p4, p5


class DetectionHead(nn.Module):
    """
    Cabeza de detección mejorada para una escala específica
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Más capacidad en la cabeza de detección
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Predicción final: num_anchors * (5 + num_classes)
        # 5 = objectness + cx + cy + w + h
        self.pred = nn.Conv2d(in_channels // 2, num_anchors * (5 + num_classes), 1)
    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        # Forzar FP32 para estabilidad numérica
        x = x.float()
        x = self.conv_block(x)
        return self.pred(x)


class SGSNet(nn.Module):
    """
    Strawberry Grading System Network v2
    Red neuronal mejorada para detección multi-escala de estados de madurez de fresas
    
    Mejoras sobre v1:
    - Backbone MobileNetV3-Large (más capacidad)
    - FPN verdadero con 3 escalas de detección
    - 9 anchors totales con aspect ratios variados
    - Módulos de atención CBAM
    """
    def __init__(self, num_classes=5, anchors_per_scale=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        self.num_anchors = anchors_per_scale  # Para compatibilidad
        self.num_scales = 3  # P3, P4, P5
        
        # Backbone: MobileNetV3-Large (más robusto que Small)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.backbone = mobilenet.features
        
        # Canales en diferentes capas de MobileNetV3-Large:
        # features[4]: 40 channels, stride 8
        # features[7]: 80 channels, stride 16  
        # features[12]: 112 channels, stride 16
        # features[16]: 960 channels, stride 32
        
        self.c3_channels = 40   # stride 8 → 64x64 para input 512
        self.c4_channels = 112  # stride 16 → 32x32
        self.c5_channels = 960  # stride 32 → 16x16
        
        # FPN
        self.fpn = FPN(
            in_channels_list=[self.c3_channels, self.c4_channels, self.c5_channels],
            out_channels=256
        )
        
        # Detection heads - uno por escala
        self.head_p3 = DetectionHead(256, anchors_per_scale, num_classes)  # 64x64
        self.head_p4 = DetectionHead(256, anchors_per_scale, num_classes)  # 32x32
        self.head_p5 = DetectionHead(256, anchors_per_scale, num_classes)  # 16x16
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inicializa los pesos de las capas no pre-entrenadas"""
        for m in [self.fpn, self.head_p3, self.head_p4, self.head_p5]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def _extract_features(self, x):
        """
        Extrae features de múltiples capas del backbone
        
        Returns:
            c3, c4, c5: Features en diferentes resoluciones
        """
        c3 = c4 = c5 = None
        
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            
            # C3: después de layer 4 (stride 8, 40 channels)
            if idx == 4:
                c3 = x
            # C4: después de layer 12 (stride 16, 112 channels)
            elif idx == 12:
                c4 = x
            # C5: después de layer 16 (stride 32, 960 channels)
            elif idx == 16:
                c5 = x
                
        return c3, c4, c5
    
    def forward(self, x):
        """
        Forward pass del modelo
        
        Args:
            x: Tensor de entrada [B, 3, H, W]
            
        Returns:
            predictions: Lista de predicciones [pred_p3, pred_p4, pred_p5]
                        Cada predicción tiene shape [B, anchors*(5+num_classes), H', W']
        """
        # Extraer features multi-escala del backbone
        c3, c4, c5 = self._extract_features(x)
        
        # FPN: generar features piramidales
        p3, p4, p5 = self.fpn(c3, c4, c5)
        
        # Aplicar detection heads
        pred_p3 = self.head_p3(p3)  # [B, A*(5+C), 64, 64]
        pred_p4 = self.head_p4(p4)  # [B, A*(5+C), 32, 32]
        pred_p5 = self.head_p5(p5)  # [B, A*(5+C), 16, 16]
        
        return [pred_p3, pred_p4, pred_p5]
    
    def get_num_params(self):
        """Retorna el número de parámetros del modelo"""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Retorna el número de parámetros entrenables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=5, pretrained_backbone=True):
    """
    Factory function para crear el modelo
    
    Args:
        num_classes: Número de clases a detectar
        pretrained_backbone: Si usar pesos pre-entrenados en backbone
        
    Returns:
        model: Instancia de SGSNet
    """
    model = SGSNet(num_classes=num_classes)
    
    total_params = model.get_num_params() / 1e6
    trainable_params = model.get_num_trainable_params() / 1e6
    
    print(f"SGSNet v2 creado:")
    print(f"  - Parámetros totales: {total_params:.2f}M")
    print(f"  - Parámetros entrenables: {trainable_params:.2f}M")
    print(f"  - Escalas de detección: 3 (P3: 64x64, P4: 32x32, P5: 16x16)")
    print(f"  - Anchors por escala: 3")
    print(f"  - Total anchors: 9")
    
    return model


# Para compatibilidad con código existente
def get_model(num_classes=5):
    """Función de compatibilidad"""
    return create_model(num_classes)
