import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
import tempfile
import os

from training.model import DeepfakeDetector
from inference.visualization import compute_gradcam, draw_result, default_preprocess
from training.data_utils import sample_k_indices

DEEPFAKE_MODEL_PATH = "deepfake_detector.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_MODEL_ID = "google/gemma-3-4b-it"

print("Loading models...")

deepfake_model = DeepfakeDetector(pretrained=True)
state_dict = torch.load(DEEPFAKE_MODEL_PATH, map_location=DEVICE)
deepfake_model.load_state_dict(state_dict)
deepfake_model = deepfake_model.to(DEVICE)
deepfake_model.eval()
print("Deepfake detector loaded")

vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
vlm_model = Gemma3ForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).eval()
print("VLM loaded")


def generate_explanation(pil_img, cx, cy, radius):
    prompt_text = (
        f"This image shows a potential deepfake. In the region at "
        f"coordinates ({int(cx)}, {int(cy)}) with radius {int(radius)} pixels, "
        f"describe in ONE short sentence what visual artifact suggests manipulation."
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You must respond only with what the user asks and in short concise sentences."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    inputs = vlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(vlm_model.device, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.inference_mode():
        generation = vlm_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    explanation = vlm_processor.decode(generation, skip_special_tokens=True).strip()
    return explanation


def analyze_image_with_vlm(pil_img):
    prob, cx, cy, radius = compute_gradcam(deepfake_model, pil_img, DEVICE)
    
    if prob <= 0.5:
        out_img = draw_result(pil_img, prob, cx, cy, radius)
        return out_img, prob, None
    
    explanation = generate_explanation(pil_img, cx, cy, radius)
    annotated_img = draw_result(pil_img, prob, cx, cy, radius, explanation)
    
    return annotated_img, prob, explanation


def extract_frames_with_probs(video_path, kframes=16):
    deepfake_model.eval()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame.copy())
    cap.release()
    
    n_frames = len(frames_bgr)
    if n_frames == 0:
        raise RuntimeError("No frames read")
    
    indices = sample_k_indices(n_frames, kframes)
    
    imgs = []
    pil_frames = []
    for i in indices:
        f = frames_bgr[i]
        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(f_rgb)
        x = default_preprocess(pil)
        imgs.append(x.squeeze(0))
        pil_frames.append(pil)
    
    batch = torch.stack(imgs, dim=0).to(DEVICE)
    
    with torch.no_grad():
        logits = deepfake_model(batch)
        if logits.dim() == 1:
            logits = logits.view(-1, 1)
        
        mean_logit = logits.mean(dim=0)
        overall_prob = torch.sigmoid(mean_logit).item()
        per_frame_probs = torch.sigmoid(logits).cpu().numpy().ravel().tolist()
    
    print(f"Video probability: {overall_prob:.4f} ({len(indices)} frames)")
    
    frames_data = [(pil_frames[i], per_frame_probs[i], indices[i]) 
                   for i in range(len(indices))]
    
    return overall_prob, frames_data


app = FastAPI(title="Deepfake Detection API")

@app.get("/")
def read_root():
    return {
        "message": "Deepfake Detection API",
        "model": VLM_MODEL_ID,
        "device": DEVICE,
        "endpoints": {
            "/detect": "POST - Upload video",
            "/detect-image": "POST - Upload image"
        }
    }

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...), kframes: int = 16):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    is_video = False
    
    if file.content_type and file.content_type.startswith("video/"):
        is_video = True
    elif file.filename and any(file.filename.lower().endswith(ext) for ext in video_extensions):
        is_video = True
    
    if not is_video:
        raise HTTPException(400, f"File must be video")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        content = await file.read()
        tmp_video.write(content)
        tmp_video_path = tmp_video.name
    
    try:
        overall_prob, frames_data = extract_frames_with_probs(tmp_video_path, kframes=kframes)
        
        if not frames_data:
            raise HTTPException(400, "Could not extract frames")
        
        if overall_prob <= 0.5:
            first_frame = frames_data[0][0]
            out_img = first_frame.convert("RGB")
            
            img_byte_arr = io.BytesIO()
            out_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return StreamingResponse(
                img_byte_arr,
                media_type="image/png",
                headers={
                    "X-Overall-Probability": str(overall_prob),
                    "X-Deepfake-Detected": "false",
                    "X-Frames-Analyzed": str(len(frames_data))
                }
            )
        
        max_frame, max_frame_prob, max_idx = max(frames_data, key=lambda x: x[1])
        
        annotated_img, _, explanation = analyze_image_with_vlm(max_frame)
        
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Overall-Probability": str(overall_prob),
                "X-Deepfake-Detected": "true",
                "X-Max-Frame-Index": str(max_idx),
                "X-Max-Frame-Probability": str(max_frame_prob),
                "X-Frames-Analyzed": str(len(frames_data)),
                "X-Gemma-Explanation": explanation
            }
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(500, f"Error: {str(e)}\n{error_details}")
    
    finally:
        if os.path.exists(tmp_video_path):
            os.unlink(tmp_video_path)

@app.post("/detect-image")
async def detect_deepfake_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be image")
    
    try:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        
        annotated_img, prob, _ = analyze_image_with_vlm(pil_img)
        
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"X-Deepfake-Probability": str(prob)}
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)