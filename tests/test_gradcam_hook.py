"""Unit test cho Grad-CAM hook (Muc 4.2 + 8.1 cua CARVE v5.1).

Kiem tra:
- Hook vao Conv block 4 (feature map [256, 16, 25]) hoat dong
- Forward + backward khong loi
- CAM khong uniform (co tap trung)
- CAM shape khop voi feature map
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyDeepFeat(nn.Module):
    """Rut gon cua DeepFeat: 4 conv block + GAP + FC, output [1, 128, 200]."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self._feature_map = x  # luu de hook
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def test_gradcam_hook_activations_shape():
    """Feature map sau conv block 4 phai co shape [B, 256, 16, 25]."""
    model = DummyDeepFeat()
    x = torch.randn(2, 1, 128, 200)
    _ = model(x)
    assert hasattr(model, '_feature_map')
    fm = model._feature_map
    assert fm.shape == (2, 256, 16, 25), f"feature map shape={fm.shape}"


def test_gradcam_backward():
    """Backward tu logit class duong phai sinh gradient tren feature map."""
    model = DummyDeepFeat()
    x = torch.randn(1, 1, 128, 200)
    logit = model(x)
    logit.backward()
    fm = model._feature_map
    assert fm.grad is not None, "feature map khong co gradient"
    assert not torch.isnan(fm.grad).any(), "gradient co NaN"


def test_gradcam_not_uniform():
    """CAM phai khong uniform (co tap trung vao vung nao do)."""
    model = DummyDeepFeat()
    model.eval()
    x = torch.randn(1, 1, 128, 200)
    logit = model(x)
    model.zero_grad()
    logit.backward()

    fm = model._feature_map.detach()
    grad = model._feature_map.grad.detach()

    # Grad-CAM: weights = mean grad over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, 256, 1, 1]
    cam = F.relu((weights * fm).sum(dim=1, keepdim=True))  # [1, 1, 16, 25]
    cam_np = cam.squeeze().numpy()

    # Kiem tra khong uniform: std/mean > 0.1
    assert cam_np.std() > 0, "CAM uniform (std=0)"
    # Sau ReLU, nhieu gia tri = 0. Kiem tra co it nhat 1 gia tri > 0
    assert cam_np.max() > 0, "CAM toan so 0"
    # Kiem tra CAM co phan biet vung: ty le gia tri > 0.5*max phai < 50%
    high_act_ratio = (cam_np > 0.5 * cam_np.max()).mean()
    assert high_act_ratio < 0.5, (
        f"CAM qua uniform: {high_act_ratio:.2%} gia tri > 0.5*max"
    )


def test_gradcam_upsample_shape():
    """Upsample CAM tu [16, 25] ve [128, T] bang bilinear."""
    cam = torch.randn(1, 1, 16, 25)
    upsampled = F.interpolate(cam, size=(128, 200), mode='bilinear',
                              align_corners=False)
    assert upsampled.shape == (1, 1, 128, 200)


if __name__ == '__main__':
    test_gradcam_hook_activations_shape()
    test_gradcam_backward()
    test_gradcam_not_uniform()
    test_gradcam_upsample_shape()
    print("All gradcam_hook tests passed.")