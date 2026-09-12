"""CARVE — DeepFeat model (VGG-style CNN, ~521k params).

Kien truc theo Muc 4.2 CARVE v5.1:
- 4 conv blocks, moi block: Conv2d + BN + ReLU (+ MaxPool cho 3 block dau)
- Global Average Pool -> Dropout(0.5) -> FC(256->512->1)
- Grad-CAM target: feature map sau Conv block 4, shape [B, 256, 16, 25]

Deviation E.6: FC head mo rong tu Linear(256,1) thanh 256->512->1,
params tang tu 389k len 521k.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFeat(nn.Module):
    """VGG-style CNN nho cho mel-spectrogram [1, 128, 200]."""

    def __init__(self, dropout=0.5):
        super().__init__()

        # Conv block 1: [1, 128, 200] -> [32, 64, 100]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 2: [32, 64, 100] -> [64, 32, 50]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 3: [64, 32, 50] -> [128, 16, 25]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Conv block 4: [128, 16, 25] -> [256, 16, 25] (khong pool)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # FC head mo rong (E.6): 256 -> 512 -> 1
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

        # Luu feature map de Grad-CAM lay
        self._feature_map = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Retain grad cho Grad-CAM (non-leaf tensor)
        if x.requires_grad:
            x.retain_grad()
        self._feature_map = x

        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # logits, khong sigmoid (dung BCEWithLogitsLoss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradcam_from_features(feature_map, grad_outputs=None):
    """Tinh Grad-CAM tu feature_map [B, C, H, W] da co .grad.

    Neu grad_outputs=None, dung .grad da co san tu backward.
    Tra ve CAM [B, 1, H, W] da ReLU va normalize [0, 1] moi sample.
    """
    if feature_map.grad is None:
        raise RuntimeError("feature_map.grad is None. Goi backward() truoc.")
    grad = feature_map.grad.detach()
    fm = feature_map.detach()

    # Weights = mean grad over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = F.relu((weights * fm).sum(dim=1, keepdim=True))  # [B, 1, H, W]

    # Normalize moi sample ve [0, 1]
    B = cam.size(0)
    cam_flat = cam.view(B, -1)
    cam_min = cam_flat.min(dim=1, keepdim=True)[0]
    cam_max = cam_flat.max(dim=1, keepdim=True)[0]
    denom = (cam_max - cam_min).clamp(min=1e-8)
    cam = (cam_flat - cam_min) / denom
    cam = cam.view(B, 1, feature_map.size(2), feature_map.size(3))
    return cam


if __name__ == '__main__':
    # Quick test
    model = DeepFeat()
    print(f"DeepFeat params: {count_parameters(model):,}")

    x = torch.randn(2, 1, 128, 200, requires_grad=True)
    logits = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Feature map shape: {tuple(model._feature_map.shape)}")

    # Backward + Grad-CAM
    logits.sum().backward()
    cam = gradcam_from_features(model._feature_map)
    print(f"CAM shape: {tuple(cam.shape)}")
    print(f"CAM range: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
    print("OK")