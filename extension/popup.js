let mediaRecorder;
let recordedChunks = [];
let recordedBlob = null;
let croppedBlob = null;
let cropRect = null;

const FASTAPI_SERVER_URL = 'https://your-server-url.ngrok-free.dev/detect';

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cropBtn = document.getElementById('cropBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const statusDiv = document.getElementById('status');
const resultDiv = document.getElementById('result');

document.addEventListener('DOMContentLoaded', async () => {
  const { recordedVideo, analysisResult } = await chrome.storage.local.get(['recordedVideo', 'analysisResult']);
  
  if (analysisResult) {
    showResult(analysisResult);
  } else if (recordedVideo) {
    const response = await fetch(recordedVideo);
    recordedBlob = await response.blob();
    statusDiv.textContent = 'Recording complete! Choose an option below.';
    cropBtn.style.display = 'block';
    analyzeBtn.style.display = 'block';
    resetBtn.style.display = 'block';
    startBtn.style.display = 'none';
  } else {
    resetBtn.style.display = 'block';
  }
});

startBtn.addEventListener('click', async () => {
  try {
    await chrome.storage.local.remove(['analysisResult', 'recordedVideo']);
    
    statusDiv.innerHTML = '<span class="loader"></span> Select screen to record...';
    
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { 
        displaySurface: "monitor",
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      },
      audio: false
    });

    recordedChunks = [];
    recordedBlob = null;
    croppedBlob = null;
    cropRect = null;
    
    mediaRecorder = new MediaRecorder(stream, { 
      mimeType: 'video/webm; codecs=vp9' 
    });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
      
      const reader = new FileReader();
      reader.onloadend = async () => {
        await chrome.storage.local.set({ recordedVideo: reader.result });
        statusDiv.textContent = 'Recording complete! Choose an option below.';
        cropBtn.style.display = 'block';
        analyzeBtn.style.display = 'block';
        stopBtn.style.display = 'none';
      };
      reader.readAsDataURL(recordedBlob);
    };

    mediaRecorder.start();

    statusDiv.innerHTML = '<span class="loader"></span> Recording... Click Stop button below.';
    startBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    resetBtn.style.display = 'block';

    stream.getVideoTracks()[0].onended = () => {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
      }
    };

  } catch (err) {
    let errorMsg = 'Error starting recording';
    if (err.name === 'NotAllowedError') {
      errorMsg = 'Screen sharing permission denied';
    } else if (err.name === 'NotFoundError') {
      errorMsg = 'No screen source selected';
    } else {
      errorMsg = err.message;
    }
    
    statusDiv.textContent = errorMsg;
    resetUI();
  }
});

stopBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
});

async function loadRecordedVideo() {
  const { recordedVideo } = await chrome.storage.local.get('recordedVideo');
  if (recordedVideo) {
    const response = await fetch(recordedVideo);
    recordedBlob = await response.blob();
    return true;
  }
  return false;
}

cropBtn.addEventListener('click', async () => {
  if (!recordedBlob) {
    await loadRecordedVideo();
  }
  if (!recordedBlob) return;
  
  statusDiv.textContent = 'Loading video for cropping...';
  
  const video = document.createElement('video');
  video.src = URL.createObjectURL(recordedBlob);
  video.muted = true;
  
  video.onloadedmetadata = () => {
    video.currentTime = Math.min(1, video.duration / 2);
  };
  
  video.onseeked = () => {
    const cropCanvas = document.getElementById('cropCanvas');
    const ctx = cropCanvas.getContext('2d');
    
    cropCanvas.width = video.videoWidth;
    cropCanvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    document.getElementById('cropModal').classList.add('active');
    setupCropSelection(video);
    statusDiv.textContent = 'Click and drag to select area';
  };
});

function setupCropSelection(sourceVideo) {
  const cropCanvas = document.getElementById('cropCanvas');
  const ctx = cropCanvas.getContext('2d');
  let isDrawing = false;
  let startX, startY;
  let tempCropRect = null;

  cropCanvas.onmousedown = (e) => {
    isDrawing = true;
    const rect = cropCanvas.getBoundingClientRect();
    startX = (e.clientX - rect.left) * (cropCanvas.width / rect.width);
    startY = (e.clientY - rect.top) * (cropCanvas.height / rect.height);
  };

  cropCanvas.onmousemove = (e) => {
    if (!isDrawing) return;
    
    const rect = cropCanvas.getBoundingClientRect();
    const currentX = (e.clientX - rect.left) * (cropCanvas.width / rect.width);
    const currentY = (e.clientY - rect.top) * (cropCanvas.height / rect.height);
    
    ctx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
    ctx.drawImage(sourceVideo, 0, 0);
    
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    
    tempCropRect = { x, y, width: w, height: h };
    
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
    
    ctx.clearRect(x, y, w, h);
    ctx.drawImage(sourceVideo, x, y, w, h, x, y, w, h);
    
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
  };

  cropCanvas.onmouseup = (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    cropRect = tempCropRect;
  };
}

document.getElementById('confirmCrop').addEventListener('click', async () => {
  if (!cropRect || cropRect.width < 10 || cropRect.height < 10) {
    alert('Please select a larger area!');
    return;
  }
  
  document.getElementById('cropModal').classList.remove('active');
  statusDiv.innerHTML = '<span class="loader"></span> Cropping video...';
  cropBtn.disabled = true;
  analyzeBtn.disabled = true;
  
  try {
    await cropVideo();
    statusDiv.textContent = `Video cropped to ${Math.round(cropRect.width)}px by ${Math.round(cropRect.height)}px`;
  } catch (err) {
    statusDiv.textContent = 'Cropping failed';
  } finally {
    cropBtn.disabled = false;
    analyzeBtn.disabled = false;
  }
});

document.getElementById('cancelCrop').addEventListener('click', () => {
  cropRect = null;
  document.getElementById('cropModal').classList.remove('active');
  statusDiv.textContent = 'Crop cancelled';
});

async function cropVideo() {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.src = URL.createObjectURL(recordedBlob);
    video.muted = true;
    
    video.onloadedmetadata = () => {
      const canvas = document.createElement('canvas');
      canvas.width = cropRect.width;
      canvas.height = cropRect.height;
      const context = canvas.getContext('2d');
      
      const stream = canvas.captureStream(30);
      const recorder = new MediaRecorder(stream, { 
        mimeType: 'video/webm; codecs=vp9' 
      });
      const chunks = [];
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      
      recorder.onstop = () => {
        croppedBlob = new Blob(chunks, { type: 'video/webm' });
        resolve();
      };
      
      recorder.start();
      video.play();
      
      const drawFrame = () => {
        if (video.paused || video.ended) {
          recorder.stop();
          return;
        }
        
        context.drawImage(
          video,
          cropRect.x, cropRect.y, cropRect.width, cropRect.height,
          0, 0, cropRect.width, cropRect.height
        );
        
        requestAnimationFrame(drawFrame);
      };
      
      drawFrame();
    };
    
    video.onerror = () => reject(new Error('Video loading failed'));
  });
}

analyzeBtn.addEventListener('click', async () => {
  if (!recordedBlob) {
    await loadRecordedVideo();
  }
  if (!recordedBlob) return;
  
  statusDiv.innerHTML = '<span class="loader"></span> Uploading video to server for analysis...';
  analyzeBtn.disabled = true;
  cropBtn.disabled = true;

  try {
    const formData = new FormData();
    const videoToAnalyze = croppedBlob || recordedBlob;
    formData.append('file', videoToAnalyze, 'recording.webm');

    const urlWithParams = `${FASTAPI_SERVER_URL}?kframes=8`;
    
    const response = await fetch(urlWithParams, {
      method: 'POST',
      body: formData,
      headers: {
        'ngrok-skip-browser-warning': 'true'
      }
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Server returned ${response.status}: ${errorText}`);
    }

    const imageBlob = await response.blob();
    
    const overallProbability = parseFloat(response.headers.get('X-Overall-Probability') || '0');
    const deepfakeDetected = response.headers.get('X-Deepfake-Detected') === 'true';
    const framesAnalyzed = response.headers.get('X-Frames-Analyzed');
    const maxFrameIndex = response.headers.get('X-Max-Frame-Index');
    const maxFrameProb = response.headers.get('X-Max-Frame-Probability');
    const gemmaExplanation = response.headers.get('X-Gemma-Explanation');
    
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Image = reader.result.split(',')[1];
      
      const analysisResult = {
        probability: overallProbability,
        verdict: deepfakeDetected ? 'FAKE' : 'REAL',
        image: base64Image,
        framesAnalyzed: framesAnalyzed,
        maxFrameIndex: maxFrameIndex,
        maxFrameProb: maxFrameProb,
        gemmaExplanation: gemmaExplanation
      };
      
      await chrome.storage.local.set({ analysisResult });
      showResult(analysisResult);
    };
    reader.readAsDataURL(imageBlob);

  } catch (error) {
    let errorMsg = 'Analysis failed. ';
    
    if (error.message.includes('Failed to fetch')) {
      errorMsg += 'Cannot connect to server. Check if server is running and URL is correct.';
    } else {
      errorMsg += error.message;
    }
    
    statusDiv.innerHTML = errorMsg;
  } finally {
    analyzeBtn.disabled = false;
    cropBtn.disabled = false;
  }
});

resetBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  
  await chrome.storage.local.remove(['analysisResult', 'recordedVideo']);
  
  recordedBlob = null;
  croppedBlob = null;
  cropRect = null;
  recordedChunks = [];
  
  document.getElementById('cropModal').classList.remove('active');
  
  resetUI();
});

function showResult(data) {
  stopBtn.style.display = 'none';
  startBtn.style.display = 'none';
  cropBtn.style.display = 'none';
  analyzeBtn.style.display = 'none';
  resetBtn.style.display = 'block';
  
  statusDiv.textContent = "Analysis Complete";
  resultDiv.style.display = 'block';
  
  const probPercent = (data.probability * 100).toFixed(1);
  let resultHTML = '';
  
  if (data.verdict === "FAKE") {
    resultDiv.className = 'fake';
    resultHTML = `<strong>DEEPFAKE DETECTED</strong><br>Confidence: ${probPercent}%`;
  } else {
    resultDiv.className = 'real';
    resultHTML = `<strong>LIKELY AUTHENTIC</strong><br>Fake Probability: ${probPercent}%`;
  }
  
  if (data.framesAnalyzed) {
    resultHTML += `<br><small>Frames analyzed: ${data.framesAnalyzed}</small>`;
  }
  
  if (data.maxFrameIndex && data.maxFrameProb) {
    resultHTML += `<br><small>Max detection at frame ${data.maxFrameIndex}: ${(parseFloat(data.maxFrameProb) * 100).toFixed(1)}%</small>`;
  }
  
  if (data.image) {
    resultHTML += `<br><br><img src="data:image/png;base64,${data.image}" alt="Analysis visualization" style="max-width: 100%; border-radius: 8px; margin-top: 12px;">`;
  }
  
  if (data.verdict === "FAKE" && data.gemmaExplanation) {
    resultHTML += `<br><br><div style="background: #fef2f2; padding: 12px; border-radius: 8px; border-left: 4px solid #dc2626; margin-top: 12px; text-align: left;">
      <strong style="color: #991b1b;">AI Explanation:</strong><br>
      <span style="color: #4a5568; font-size: 13px; line-height: 1.5;">${data.gemmaExplanation}</span>
    </div>`;
  }
  
  resultDiv.innerHTML = resultHTML;
}

function resetUI() {
  startBtn.style.display = 'block';
  stopBtn.style.display = 'none';
  cropBtn.style.display = 'none';
  analyzeBtn.style.display = 'none';
  resetBtn.style.display = 'block';
  resultDiv.style.display = 'none';
  statusDiv.textContent = "Ready to record";
  analyzeBtn.disabled = false;
  cropBtn.disabled = false;
}