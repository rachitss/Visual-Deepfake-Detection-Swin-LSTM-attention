import argparse
import os
import shutil
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel, logging

from extract_faces import get_frames

logging.set_verbosity_error()
MODEL_WEIGHTS = 'model.pth'


def load_model(model_path: str = MODEL_WEIGHTS, device: Optional[torch.device] = None) -> Tuple:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained('microsoft/swin-tiny-patch4-window7-224', use_fast=True)
    backbone = AutoModel.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
    hidden_size = backbone.config.hidden_size

    lstm = nn.LSTM(input_size=hidden_size, hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
    attn = nn.Linear(256, 1)
    relu = nn.ReLU()
    classifier = nn.Linear(256, 1)

    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'backbone' in state_dict:
        backbone.load_state_dict(state_dict['backbone'], strict=False)
        lstm.load_state_dict(state_dict['lstm'])
        classifier.load_state_dict(state_dict['classifier'])
        attn.load_state_dict(state_dict['attn'])
    else:
        backbone.load_state_dict(state_dict, strict=False)

    backbone = backbone.to(device).eval()
    lstm = lstm.to(device).eval()
    attn = attn.to(device).eval()
    relu = relu.to(device).eval()
    classifier = classifier.to(device).eval()

    return processor, backbone, lstm, attn, relu, classifier, device


def run_inference(
    video_path: str,
    *,
    model_path: str = MODEL_WEIGHTS,
    processor=None,
    model=None,
    lstm=None,
    attn=None,
    relu=None,
    classifier=None,
    device: Optional[torch.device] = None,
    batch_size: int = 36,
) -> dict:
    """Extract frames from the video and return prediction metadata."""

    modules = (processor, model, lstm, attn, relu, classifier)
    if any(m is None for m in modules) or device is None:
        processor, model, lstm, attn, relu, classifier, device = load_model(
            model_path=model_path, device=device
        )

    frame_dir = get_frames(video_path)
    try:
        image_files = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith('.png')
            ]
        )
        if not image_files:
            raise RuntimeError('No frames extracted from video; cannot run inference.')

        print('\n--------------Running Inference------------------')
        frame_feats = []
        for i in tqdm(range(0, len(image_files), batch_size), desc='Image Batches'):
            batch_paths = image_files[i : i + batch_size]
            images = [Image.open(p).convert('RGB') for p in batch_paths]
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False, return_dict=True)
                pooled = outputs.pooler_output

            frame_feats.append(pooled)

        if not frame_feats:
            raise RuntimeError('Failed to extract frame features; cannot run inference.')

        seq = torch.cat(frame_feats, dim=0).unsqueeze(0)

        with torch.no_grad():
            lstm_out, _ = lstm(seq)
            att_score = relu(attn(lstm_out))
            norm_score = torch.softmax(att_score, dim=1)
            context = (norm_score * lstm_out).sum(dim=1)
            logit = classifier(context)
            probability = torch.sigmoid(logit).item()

        pred = int(probability >= 0.5)
        return {
            'video_path': video_path,
            'prediction': pred,
            'frame_count': len(image_files),
            'probability': probability,
        }
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Run deepfake inference on a video file.')
    parser.add_argument('--video', required=True, help='Path to the video file to analyze')
    parser.add_argument('--model-path', default=MODEL_WEIGHTS, help='Path to model weights')
    args = parser.parse_args()

    result = run_inference(args.video, model_path=args.model_path)
    print('\n--------------Result------------------')
    print(f"Video: {result['video_path']}")
    print(f"Predicted class: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")


if __name__ == '__main__':
    main()
