from pathlib import Path
from typing import List
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        
        for label_name, label in [("real", 0), ("fake", 1)]:
            id_dirs = self._list_id_dirs(label_name, split)
            for idd in id_dirs:
                frames = self._list_frame_files(idd)
                for f in frames:
                    self.samples.append((str(f), label))
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No frames found in {root} (split='{split}')")

        real = [(p, l) for p, l in self.samples if l == 0]
        fake = [(p, l) for p, l in self.samples if l == 1][:len(real)]
        self.samples = real + fake
        random.shuffle(self.samples)

    def _list_id_dirs(self, label_name: str, split: str) -> List[Path]:
        p = self.root / label_name / split
        if not p.exists():
            return []
        return [d for d in sorted(p.iterdir()) if d.is_dir()]

    def _list_frame_files(self, video_dir: Path) -> List[Path]:
        exts = ("*.jpg", "*.jpeg", "*.png")
        files = []
        for e in exts:
            files.extend(sorted(video_dir.glob(e)))
        return files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    def __init__(self, root: str, kframes: int = 8, transform=None):
        super().__init__()
        self.root = Path(root)
        self.k = int(kframes)
        self.transform = transform
        self.samples = []

        for label_name, label in [("Celeb-real", 0), ("Celeb-synthesis", 1)]:
            video_dir = self.root / label_name
            if not video_dir.exists():
                continue
            for vp in sorted(video_dir.glob("*.mp4")):
                self.samples.append((vp, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No MP4 videos found under {root}")

        real = [(p, l) for p, l in self.samples if l == 0]
        fake = [(p, l) for p, l in self.samples if l == 1][:len(real)]
        self.samples = real + fake
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        ok = True
        while ok:
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return torch.zeros(self.k, 3, 224, 224), torch.tensor(label, dtype=torch.float32)

        indices = sample_k_indices(len(frames), self.k)
        imgs = []
        for i in indices:
            f = frames[i]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(f)
            if self.transform:
                pil = self.transform(pil)
            else:
                pil = T.ToTensor()(pil)
            imgs.append(pil)

        stack = torch.stack(imgs, dim=0)
        return stack, torch.tensor(label, dtype=torch.float32)


def sample_k_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if n >= k:
        return list(np.linspace(0, n - 1, k, dtype=int))
    idx = list(range(n))
    while len(idx) < k:
        idx.append(n - 1)
    return idx[:k]


def get_transform(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])