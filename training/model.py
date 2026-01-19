import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.efficientnet_b0(pretrained=pretrained)
        in_features = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_features, 1)
        self.model = base

    def forward(self, x):
        return self.model(x)


class VideoAggregator(nn.Module):
    def __init__(self, frame_detector):
        super().__init__()
        self.frame_detector = frame_detector

    def forward(self, frames):
        B, K, C, H, W = frames.shape
        x = frames.view(B * K, C, H, W)
        logits = self.frame_detector(x)
        if logits.dim() == 1:
            logits = logits.view(-1, 1)
        logits = logits.view(B, K, -1)
        mean_logits = logits.mean(dim=1)
        return mean_logits


def save_checkpoint(model, path):
    sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(sd, path)


def load_checkpoint(model, path, device):
    ck = torch.load(path, map_location=device)
    
    if isinstance(ck, dict):
        for key in ("state_dict", "model_state", "model", "state"):
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break

    if not isinstance(ck, dict):
        raise RuntimeError(f"Checkpoint {path} does not contain valid state dict")

    model_keys = set(model.state_dict().keys())
    ck_keys_set = set(ck.keys())
    
    if len(model_keys & ck_keys_set) >= max(1, int(0.2 * len(model_keys))):
        normalized = {}
        for k, v in ck.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            normalized[nk] = v
        model.load_state_dict(normalized, strict=False)
        return

    for prefix in ("module.frame_detector.", "frame_detector.", "module.model.", "model."):
        sub = {}
        found = False
        for k, v in ck.items():
            if k.startswith(prefix):
                found = True
                newk = k[len(prefix):]
                sub[newk] = v
        if found:
            normalized = {}
            for k, v in sub.items():
                nk = k.replace("module.", "") if k.startswith("module.") else k
                normalized[nk] = v
            model.load_state_dict(normalized, strict=False)
            return

    normalized = {}
    for k, v in ck.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        normalized[nk] = v
    model.load_state_dict(normalized, strict=False)