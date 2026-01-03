"""FastAPI service exposing the deepfake inference pipeline."""
import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from infer import load_model, run_inference

app = FastAPI(title="Deepfake Detection API", version="0.1.0")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(
    PROCESSOR,
    MODEL,
    LSTM,
    ATTN,
    RELU,
    CLASSIFIER,
    DEVICE,
) = load_model(device=DEVICE)
CHUNK_SIZE = 1024 * 1024  # 1 MB


@app.get('/health')
def healthcheck() -> dict:
    return {'status': 'ok', 'device': str(DEVICE)}


@app.post('/inference')
async def inference_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Uploaded file must have a name.')

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {'.mp4', '.mov', '.avi', '.mkv'}:
        raise HTTPException(status_code=400, detail='Unsupported file type. Upload a video file.')

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        result = run_inference(
            tmp_path,
            processor=PROCESSOR,
            model=MODEL,
            lstm=LSTM,
            attn=ATTN,
            relu=RELU,
            classifier=CLASSIFIER,
            device=DEVICE,
        )
        return JSONResponse(result)
    finally:
        os.remove(tmp_path)
