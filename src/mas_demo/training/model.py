from torch import nn
import torch
import hashlib
from importlib.resources import files
import numpy as np


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 128):
            layers.extend(
                [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            )
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def _blobs_asset():
    return files("mas_demo").joinpath("training/assets/autotrainer_blobs_initial.npz")


class BlobsMLP(nn.Module):
    """Original 2->50 ReLU->3 model; CrossEntropyLoss fuses the output softmax."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 3))
        with _blobs_asset().open("rb") as stream, np.load(
            stream, allow_pickle=False
        ) as weights:
            with torch.no_grad():
                self.layers[0].weight.copy_(torch.from_numpy(weights["hidden_weight"]))
                self.layers[0].bias.copy_(torch.from_numpy(weights["hidden_bias"]))
                self.layers[2].weight.copy_(torch.from_numpy(weights["output_weight"]))
                self.layers[2].bias.copy_(torch.from_numpy(weights["output_bias"]))

    def forward(self, x):
        return self.layers(x)


def create_model(spec):
    return BlobsMLP() if spec.dataset == "blobs" else SmallCNN()


def model_signature(spec):
    if spec.dataset != "blobs":
        return "SmallCNN"
    digest = hashlib.sha256(_blobs_asset().read_bytes()).hexdigest()
    return f"BlobsMLP-v1:{digest}"


def task_description(spec):
    if spec.dataset == "blobs":
        return "Blobs classification / BlobsMLP (2 input features, 50 ReLU units, 3 classes)"
    return f"{spec.dataset} classification / SmallCNN (3x32x32 images, 10 classes)"
