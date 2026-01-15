import torch
import torch.nn as nn

class TwoHeadGestureModel(nn.Module):
    def __init__(self, num_count_classes: int = 12, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slow_r50',
            pretrained=pretrained
        )

        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Identity()

        self.detection_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

        self.counting_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_count_classes)
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=[2, 3, 4])
        detection = self.detection_head(features).squeeze(-1)
        count_logits = self.counting_head(features)
        return detection, count_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        detection, count_logits = self.forward(x)
        counts = count_logits.argmax(dim=-1)
        counts = torch.where(detection > 0.5, counts, torch.zeros_like(counts))
        return counts


def create_model(num_count_classes: int = 12, pretrained: bool = True, dropout: float = 0.3):
    return TwoHeadGestureModel(
        num_count_classes=num_count_classes,
        pretrained=pretrained,
        dropout=dropout
    )
