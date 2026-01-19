import os
import glob
import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

from training.model import DeepfakeDetector, VideoAggregator, load_checkpoint
from training.data_utils import sample_k_indices, IMAGENET_MEAN, IMAGENET_STD


def get_latest_checkpoint(pattern="*.pt"):
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found matching {pattern}")
    return max(checkpoints, key=os.path.getmtime)


def predict_video(video_path, weights_path=None, kframes=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(video_path):
        raise RuntimeError(f"Video not found: {video_path}")

    if weights_path is None:
        weights_path = get_latest_checkpoint()
    elif not os.path.exists(weights_path):
        raise RuntimeError(f"Weights file not found: {weights_path}")

    frame_model = DeepfakeDetector(pretrained=False)
    frame_model = frame_model.to(device)
    load_checkpoint(frame_model, weights_path, device)
    frame_model.eval()

    wrapper = VideoAggregator(frame_model)
    wrapper = wrapper.to(device)
    
    if torch.cuda.device_count() > 1:
        wrapper = nn.DataParallel(wrapper)

    wrapper.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame.copy())
    cap.release()
    
    n_frames = len(frames_bgr)
    if n_frames == 0:
        raise RuntimeError("No frames read from video")

    indices = sample_k_indices(n_frames, kframes)
    imgs = []
    for i in indices:
        f = frames_bgr[i]
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(f_rgb)
        x = transform(pil)
        imgs.append(x)

    batch_frames = torch.stack(imgs, dim=0).to(device)
    
    with torch.no_grad():
        per_frame_logits = frame_model(batch_frames)
        if per_frame_logits.dim() == 1:
            per_frame_logits = per_frame_logits.view(-1, 1)
        per_frame_probs = torch.sigmoid(per_frame_logits).cpu().numpy().ravel().tolist()

    batch_for_wrapper = batch_frames.unsqueeze(0)
    with torch.no_grad():
        video_logits = wrapper(batch_for_wrapper)
        if video_logits.dim() == 1:
            video_logits = video_logits.view(1, 1)
        video_logit = video_logits.squeeze().item()
        video_prob = float(torch.sigmoid(torch.tensor(video_logit)).item())

    print(f"Video probability: {video_prob:.4f} ({len(indices)} frames)")
    print("Per-frame probabilities:")
    for idx, p in zip(indices, per_frame_probs):
        print(f"  frame {idx}: {p:.4f}")

    return video_prob, indices, per_frame_probs