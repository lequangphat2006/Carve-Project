"""Unit test cho Grad-CAM hook."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyDeepFeat(nn.Module):
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
        x.retain_grad()
        self._feature_map = x
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def test_gradcam_hook_activations_shape():
    model = DummyDeepFeat()
    x = torch.randn(2, 1, 128, 200)
    _ = model(x)
    assert hasattr(model, '_feature_map')
    assert model._feature_map.shape == (2, 256, 16, 25)


def test_gradcam_backward():
    model = DummyDeepFeat()
    x = torch.randn(1, 1, 128, 200)
    logit = model(x)
    logit.backward()
    fm = model._feature_map
    assert fm.grad is not None
    assert not torch.isnan(fm.grad).any()


def test_gradcam_not_uniform():
    model = DummyDeepFeat()
    model.eval()
    x = torch.randn(1, 1, 128, 200)
    logit = model(x)
    model.zero_grad()
    logit.backward()
    fm = model._feature_map.detach()
    grad = model._feature_map.grad.detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fm).sum(dim=1, keepdim=True))
    cam_np = cam.squeeze().numpy()
    assert cam_np.std() > 0
    assert cam_np.max() > 0
    high_act_ratio = (cam_np > 0.5 * cam_np.max()).mean()
    assert high_act_ratio < 0.5


def test_gradcam_upsample_shape():
    cam = torch.randn(1, 1, 16, 25)
    upsampled = F.interpolate(cam, size=(128, 200), mode='bilinear',
                              align_corners=False)
    assert upsampled.shape == (1, 1, 128, 200)
