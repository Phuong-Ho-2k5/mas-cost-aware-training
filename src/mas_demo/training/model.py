from pathlib import Path
import json
from torch import nn
import torch
import hashlib
from importlib.resources import files
import numpy as np


class Exponential(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class KerasFlatten(nn.Module):
    """Flatten matching Keras channels_last ordering (N, C, H, W) -> (N, H*W*C)."""

    def forward(self, x):
        if x.ndim == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
        return x.flatten(1)


def _make_activation(name):
    if name in (None, "linear"):
        return None
    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "softsign": nn.Softsign,
        "exponential": Exponential,
        "softmax": lambda: nn.Softmax(dim=1),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name]()


def _apply_weight_init(module, mode):
    if mode in (None, "keep"):
        return
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if mode == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif mode == "kaiming":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif mode == "lecun":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class SequentialSpecModel(nn.Module):
    """Build a PyTorch sequential model from a converted AUTOTRAINER spec + .npz weights."""

    def __init__(
        self,
        spec_path,
        weights_path=None,
        hidden_activation="keep",
        use_batchnorm=False,
        weight_init="keep",
    ):
        super().__init__()
        self.spec_data = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        spec_layers = self.spec_data["layers"]
        param_indices = [
            i for i, e in enumerate(spec_layers) if e["type"] in ("Dense", "Conv2D")
        ]
        last_param_idx = param_indices[-1] if param_indices else len(spec_layers) - 1
        modules = []
        weight_layers = []
        for idx, entry in enumerate(spec_layers):
            kind = entry["type"]
            is_hidden = idx < last_param_idx
            if kind == "Dense":
                layer = nn.Linear(
                    entry["in_features"],
                    entry["out_features"],
                    bias=entry.get("use_bias", True),
                )
                modules.append(layer)
                weight_layers.append((entry["weight_key"], layer))
                if use_batchnorm and is_hidden:
                    modules.append(nn.BatchNorm1d(entry["out_features"]))
            elif kind == "Conv2D":
                layer = nn.Conv2d(
                    entry["in_channels"],
                    entry["out_channels"],
                    kernel_size=tuple(entry["kernel_size"]),
                    stride=tuple(entry.get("strides", [1, 1])),
                    padding=entry.get("padding", 0),
                    bias=entry.get("use_bias", True),
                )
                modules.append(layer)
                weight_layers.append((entry["weight_key"], layer))
                if use_batchnorm and is_hidden:
                    modules.append(nn.BatchNorm2d(entry["out_channels"]))
            elif kind == "MaxPooling2D":
                modules.append(
                    nn.MaxPool2d(
                        kernel_size=tuple(entry["pool_size"]),
                        stride=tuple(entry.get("strides", entry["pool_size"])),
                    )
                )
            elif kind == "AveragePooling2D":
                modules.append(
                    nn.AvgPool2d(
                        kernel_size=tuple(entry["pool_size"]),
                        stride=tuple(entry.get("strides", entry["pool_size"])),
                    )
                )
            elif kind == "Flatten":
                modules.append(KerasFlatten())
            elif kind == "Dropout":
                modules.append(nn.Dropout(p=float(entry.get("rate", 0.0))))
            elif kind == "Activation":
                act_name = entry.get("activation")
                if is_hidden and hidden_activation != "keep" and act_name not in (None, "linear"):
                    act_name = hidden_activation
                act = _make_activation(act_name)
                if act is not None:
                    modules.append(act)
            else:
                raise ValueError(f"Unsupported spec layer type: {kind}")
            if kind in ("Dense", "Conv2D"):
                act_name = entry.get("activation")
                if is_hidden and hidden_activation != "keep" and act_name not in (None, "linear"):
                    act_name = hidden_activation
                act = _make_activation(act_name)
                if act is not None:
                    modules.append(act)
        self.layers = nn.Sequential(*modules)
        if weight_init != "keep":
            _apply_weight_init(self, weight_init)
        elif weights_path:
            with np.load(weights_path, allow_pickle=False) as weights:
                with torch.no_grad():
                    for key, layer in weight_layers:
                        layer.weight.copy_(torch.from_numpy(weights[f"{key}__weight"]))
                        if layer.bias is not None and f"{key}__bias" in weights:
                            layer.bias.copy_(torch.from_numpy(weights[f"{key}__bias"]))

    def load_compatible_state(self, state_dict):
        """Copy Linear/Conv2d weights in layer order even if BatchNorm layers were added/removed."""
        try:
            self.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass
        # Extract (weight, bias) pairs for Linear/Conv2d from old state_dict in index order.
        old_params = []
        seen = set()
        for key in state_dict:
            if key.endswith(".weight"):
                prefix = key[:-7]
                if prefix in seen:
                    continue
                tensor = state_dict[key]
                # BatchNorm weights are 1D; Linear is 2D, Conv2d is 4D.
                if tensor.ndim in (2, 4):
                    seen.add(prefix)
                    old_params.append((tensor, state_dict.get(f"{prefix}.bias")))
        new_layers = [m for m in self.layers if isinstance(m, (nn.Linear, nn.Conv2d))]
        with torch.no_grad():
            for layer, (w, b) in zip(new_layers, old_params):
                if layer.weight.shape == w.shape:
                    layer.weight.copy_(w)
                if layer.bias is not None and b is not None and layer.bias.shape == b.shape:
                    layer.bias.copy_(b)

    def forward(self, x):
        return self.layers(x)


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


def create_model(spec, config=None):
    cfg = config or spec.training
    if getattr(spec, "model_spec_path", None):
        return SequentialSpecModel(
            spec.model_spec_path,
            spec.initial_weights_path,
            hidden_activation=getattr(cfg, "hidden_activation", "keep"),
            use_batchnorm=bool(getattr(cfg, "use_batchnorm", False)),
            weight_init=getattr(cfg, "weight_init", "keep"),
        )
    model = BlobsMLP() if spec.dataset == "blobs" else SmallCNN()
    if getattr(cfg, "weight_init", "keep") != "keep":
        _apply_weight_init(model, cfg.weight_init)
    return model


def model_summary(spec, config=None):
    """Compact objective architecture summary for Level-2 diagnosis and planning."""
    cfg = config or spec.training
    hidden_act_override = getattr(cfg, "hidden_activation", "keep")
    use_bn = bool(getattr(cfg, "use_batchnorm", False))
    if not getattr(spec, "model_spec_path", None):
        return {
            "architecture_type": "MLP" if spec.dataset == "blobs" else "CNN",
            "num_hidden_layers": 1 if spec.dataset == "blobs" else 3,
            "hidden_activations": {"relu": 1 if spec.dataset == "blobs" else 3},
            "has_batchnorm": use_bn,
            "exploding_activation": False,
            "vanishing_activation": False,
        }
    data = json.loads(Path(spec.model_spec_path).read_text(encoding="utf-8"))
    spec_layers = data.get("layers", [])
    param_indices = [
        i for i, e in enumerate(spec_layers) if e["type"] in ("Dense", "Conv2D")
    ]
    last_param_idx = param_indices[-1] if param_indices else len(spec_layers) - 1
    arch = "CNN" if any(e["type"] == "Conv2D" for e in spec_layers) else "MLP"
    hidden_acts = {}
    num_hidden = 0
    for idx, entry in enumerate(spec_layers):
        if idx >= last_param_idx:
            continue
        if entry["type"] in ("Dense", "Conv2D"):
            num_hidden += 1
        act = entry.get("activation")
        if act not in (None, "linear"):
            eff = hidden_act_override if hidden_act_override != "keep" else act
            hidden_acts[eff] = hidden_acts.get(eff, 0) + 1
    exploding = hidden_acts.get("exponential", 0) > 0
    vanishing = (hidden_acts.get("sigmoid", 0) + hidden_acts.get("tanh", 0)) >= 4 and not use_bn
    return {
        "architecture_type": arch,
        "num_hidden_layers": num_hidden,
        "hidden_activations": hidden_acts,
        "has_batchnorm": use_bn,
        "exploding_activation": exploding,
        "vanishing_activation": vanishing,
    }


def model_signature(spec):
    if getattr(spec, "model_spec_path", None):
        spec_digest = hashlib.sha256(Path(spec.model_spec_path).read_bytes()).hexdigest()
        weight_digest = (
            hashlib.sha256(Path(spec.initial_weights_path).read_bytes()).hexdigest()
            if spec.initial_weights_path
            else "none"
        )
        return f"SequentialSpecModel-v1:{spec_digest}:{weight_digest}"
    if spec.dataset != "blobs":
        return "SmallCNN"
    digest = hashlib.sha256(_blobs_asset().read_bytes()).hexdigest()
    return f"BlobsMLP-v1:{digest}"


def task_description(spec):
    if getattr(spec, "model_spec_path", None):
        data = json.loads(Path(spec.model_spec_path).read_text(encoding="utf-8"))
        return data.get("description") or f"{spec.dataset} classification / SequentialSpecModel"
    if spec.dataset == "blobs":
        return "Blobs classification / BlobsMLP (2 input features, 50 ReLU units, 3 classes)"
    return f"{spec.dataset} classification / SmallCNN (3x32x32 images, 10 classes)"
