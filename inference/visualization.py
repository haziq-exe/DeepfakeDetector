import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def default_preprocess(img_pil, input_size=224):
    tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return tf(img_pil).unsqueeze(0)


def find_last_conv(model):
    last_mod = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            last_mod = mod
    return last_mod


def compute_gradcam(model, pil_img, device, preprocess=None, input_size=224, target_layer=None):
    model.eval()
    
    if preprocess is None:
        preprocess = lambda im: default_preprocess(im, input_size=input_size)

    if target_layer is None:
        target_layer = find_last_conv(model)
    
    if target_layer is None:
        raise ValueError("Could not find convolutional layer")
        
    activations, gradients = {}, {}
    
    def f_hook(m, i, o):
        activations['value'] = o.detach()
    
    def b_hook(m, gi, go):
        gradients['value'] = go[0].detach()
    
    fh = target_layer.register_forward_hook(f_hook)
    bh = target_layer.register_full_backward_hook(b_hook)

    x = preprocess(pil_img).to(device)
    x.requires_grad = True
    
    with torch.enable_grad():
        out = model(x)
        score = out.reshape(-1)[0]
        prob = torch.sigmoid(score).item()
        model.zero_grad()
        score.backward()
    
    fh.remove()
    bh.remove()

    act = activations['value']
    grad = gradients['value']
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = cam
    
    cam_img = Image.fromarray(np.uint8(cam_norm * 255)).resize(
        pil_img.size, resample=Image.BILINEAR
    )
    cam_arr = np.array(cam_img) / 255.0

    thresh_val = np.percentile(cam_arr.flatten(), 70)
    ys, xs = np.where(cam_arr >= thresh_val)
    
    if len(xs) == 0:
        centroid_y, centroid_x = np.unravel_index(cam_arr.argmax(), cam_arr.shape)
        radius = max(pil_img.size) * 0.04
    else:
        centroid_x, centroid_y = xs.mean(), ys.mean()
        dists = np.sqrt((xs - centroid_x)**2 + (ys - centroid_y)**2)
        radius = np.percentile(dists, 60)
        radius = max(radius * 0.7, 8)

    return prob, float(centroid_x), float(centroid_y), float(radius)


def draw_result(pil_img, prob, cx, cy, radius, explanation=None, circle_width=6):
    W, H = pil_img.size
    text_height = 80
    final_canvas = Image.new("RGB", (W, H + text_height), (20, 20, 20))
    final_canvas.paste(pil_img, (0, 0))
    
    draw = ImageDraw.Draw(final_canvas)
    
    if prob > 0.5:
        circle_color = (255, 255, 0) if prob <= 0.75 else (255, 0, 0)
        
        for w in range(circle_width):
            draw.ellipse(
                [cx - radius - w, cy - radius - w, cx + radius + w, cy + radius + w],
                outline=circle_color
            )
    else:
        circle_color = (0, 255, 0)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    if prob > 0.5 and explanation:
        draw.text((10, H + 10), f"AI: {explanation}", fill=(255, 255, 255), font=font)
        draw.text((10, H + 40), f"Probability: {prob:.2%}", fill=circle_color, font=font)
    elif prob > 0.5:
        draw.text((10, H + 10), f"Deepfake Probability: {prob:.2%}", fill=circle_color, font=font)
    else:
        draw.text((10, H + 10), f"No deepfake detected (Prob: {prob:.2%})", fill=circle_color, font=font)
    
    return final_canvas