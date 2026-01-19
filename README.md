<div align="center">

# Deepfake Detection System

<br>
</div>

<p align="center">
  <img src="ReadmeExample.gif" alt="" width="900"/>
</p>

A real-time deepfake detection system that combines deep learning with explainable AI. This project won 2nd place at the InnovateX university-wide hackathon.

## Overview

This system detects manipulated videos using a two-stage neural network approach and provides visual explanations for its predictions. Users can analyze videos through a Chrome extension that communicates with a FastAPI backend server that can be run on a google colab notebook.

## Architecture

### Model Design

The detection system uses a hierarchical approach:

**Stage 1: Frame-Level Detection**
- Base model: EfficientNet-B0 pretrained on ImageNet
- Modified final layer to output single logit for binary classification
- Trained on individual frames extracted from deepfake and authentic videos
- Input: 224x224 RGB images normalized with ImageNet statistics

**Stage 2: Video-Level Aggregation**
- Wraps the frame detector to process temporal sequences
- Samples K frames uniformly from input video
- Runs frame detector on each sampled frame independently
- Aggregates frame-level predictions by averaging logits
- Final video-level prediction computed via sigmoid of mean logit

### Training Strategy

The training occurs in two phases:

1. **Frame Training**: The frame detector is trained on static images with binary cross-entropy loss. This teaches the model to identify artifacts in individual frames.

2. **Video Training**: The video aggregator is fine-tuned on complete videos with video-level labels. This refines the temporal aggregation and improves generalization to real-world video sequences.

Class imbalance is handled using weighted loss functions that adjust for the ratio of real to fake samples.

### Explainability: GradCAM Visualization

When a deepfake is detected, the system generates visual explanations:

**Gradient-Weighted Class Activation Mapping (GradCAM)**
- Identifies which regions of the image most influenced the prediction
- Computes gradients of the output with respect to final convolutional layer activations
- Weights activation maps by gradient importance
- Creates heatmap showing spatial regions critical to classification

**Implementation Flow:**
1. Forward pass through network to get prediction
2. Backward pass to compute gradients at target convolutional layer
3. Weight activation maps by gradient magnitudes
4. Generate normalized heatmap
5. Find centroid and radius of high-activation region
6. Draw circle overlay on original image

**Multimodal Explanation**
- High-activation region coordinates passed to Gemma 3 vision-language model
- VLM analyzes the circled region and generates natural language explanation
- Provides human-interpretable reasoning for why the region appears manipulated

## Chrome Extension

The browser extension provides a user-friendly interface for deepfake detection:

**Features:**
- Screen recording with MediaRecorder API
- Interactive video cropping tool for selecting regions of interest
- Upload to analysis server with progress indication
- Display results with visual annotations and explanations

**Workflow:**
1. User initiates screen recording of target content
2. Optional: Crop video to focus on specific region
3. Extension sends video to FastAPI server
4. Server processes video and returns annotated frame
5. Results displayed with probability score and AI explanation


## API Server

The FastAPI backend handles video processing and model inference:

**Endpoints:**

`POST /detect` - Analyze video file
- Accepts: video file, kframes parameter
- Returns: annotated image with visualization
- Headers include: overall probability, per-frame stats, AI explanation

`POST /detect-image` - Analyze single image
- Accepts: image file
- Returns: annotated image with GradCAM overlay


## Installation

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

## Training

**Frame-level training:**
```bash
python train.py --mode frame --data /path/to/frames --epochs 10 --batch-size 32
```

**Video-level training:**
```bash
python train.py --mode video --data /path/to/videos --epochs 5 --pretrained model_frame_epoch10.pt --kframes 20
```

## Inference

**Command-line prediction:**
```bash
python predict.py --video sample.mp4 --weights model_video_epoch5.pt --kframes 8
```

## Running the Server

```bash
cd server
uvicorn api:app --host 0.0.0.0 --port 8000
```

For production deployment with ngrok:
```bash
ngrok http 8000
```

Update the `FASTAPI_SERVER_URL` in `extension/popup.js` with your ngrok URL.

## Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `extension` directory
5. The extension icon will appear in your toolbar

## Dataset Format

**Frame dataset structure:**

[Dataset Source](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)

```
data/
├── real/
│   └── train/
│       ├── video1_frames/
│       │   ├── frame_001.jpg
│       │   └── frame_002.jpg
│       └── video2_frames/
└── fake/
    └── train/
        └── video1_frames/
```

**Video dataset structure:**

[Dataset Source](https://www.kaggle.com/datasets/debajyatidey/celeb-df-v2-real-videos-cropped-frames)

```
data/
├── Celeb-real/
│   ├── video1.mp4
│   └── video2.mp4
└── Celeb-synthesis/
    ├── video1.mp4
    └── video2.mp4
```