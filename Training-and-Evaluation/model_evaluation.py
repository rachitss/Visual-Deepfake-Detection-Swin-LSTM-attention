from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


model_path = '<path-to-saved-model>'
data_path = '<path-to-train-test-split>'
RESULTS_PATH = '<path-to-save-val-results>'
BATCH_SIZE = 32
MAX_IO_WORKERS = 16
PREFETCH_WORKERS = 16

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# load data
data = np.load(data_path, allow_pickle=True)
x_val = data['x_val']
y_val = data['y_val']

processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True) # image processor object
model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224") # model object
hidden_size = model.config.hidden_size  # feature size for LSTM input

# LSTM object
lstm = nn.LSTM(
    input_size=hidden_size,   # from transformer hidden size
    hidden_size=256,
    num_layers=1,
    batch_first=True,
    bidirectional=False)

attn = nn.Linear(256,1) # attention layer
relu = nn.ReLU() # ReLU activation function
classifier = nn.Linear(256, 1) # final classification layer
criterion = nn.BCEWithLogitsLoss(reduction='none')  # balanced Binary Cross Entropy Loss

# Moving model to device and setting to eval
model = model.to(device)
model.eval()
relu = relu.to(device)
relu.eval()
attn = attn.to(device)
attn.eval()
lstm = lstm.to(device)
lstm.eval()
classifier = classifier.to(device)
classifier.eval()

# load model weights
print("Loading model weights...")
state = torch.load(model_path, map_location='cpu')        

model.load_state_dict(state['backbone'], strict=False)
lstm.load_state_dict(state['lstm'])
classifier.load_state_dict(state['classifier'])
attn.load_state_dict(state['attn'])

model = model.to(device)
lstm = lstm.to(device)
classifier = classifier.to(device)
attn = attn.to(device)


def load_image(path): # function to load image
    with Image.open(path) as img:
        return img.convert("RGB")
    

def prepare_video(sample): # function to prepare video batches
    path, label = sample
    image_files = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith('.png')
    ])

    # prepare batches of images
    prepared_batches = []
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        if not batch_paths:
            continue
        with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as pool:
            images = list(pool.map(load_image, batch_paths))
        if not images:
            continue
        prepared_batches.append(
            processor(images=images, return_tensors="pt")
        )

    return path, prepared_batches, label

batch_size = BATCH_SIZE
video_paths = x_val
video_labels = y_val
total_batches = 0
val_loss = np.zeros(1)

labels = []
probabs = []  # sigmoid output per video
losses = []

# set models to eval mode
model.eval()
lstm.eval()
attn.eval()
classifier.eval()

total_videos = 0
total_images = 0
correct = 0
running_loss = 0.0
total_batches = 0
videos_processed = 0
start_time = time.time()

# shuffle data
perm = np.random.permutation(len(video_paths))
shuffled_paths = video_paths[perm]
shuffled_labels = video_labels[perm]

video_bar = tqdm(total=len(shuffled_paths), desc='Videos', ncols=100, leave=False)
video_iter = iter(zip(shuffled_paths, shuffled_labels))

# prefetching with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as prefetch_executor:
    future_queue = deque()

    def enqueue_next(): # function to enqueue next video preparation
        try:
            sample = next(video_iter)
        except StopIteration:
            return False
        future_queue.append(prefetch_executor.submit(prepare_video, sample))
        return True

    # initially enqueue videos
    for _ in range(PREFETCH_WORKERS):
        if not enqueue_next():
            break

    # process videos as they are prepared
    while future_queue:
        future = future_queue.popleft()
        path, prepared_batches, label = future.result()
        enqueue_next()

        video_run_loss = 0.0
        batches = 0
        frame_feats = []

        # process each batch of images through swin transformer
        for batch_inputs in prepared_batches:
            inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            total_images += inputs['pixel_values'].size(0)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=False, return_dict=True)
                pooler_output = outputs.pooler_output  # (B, C)
            frame_feats.append(pooler_output)

            batches += 1
            total_batches += 1

        # process sequence through LSTM, attention layer and classifier
        seq = torch.cat(frame_feats, dim=0)
        seq = seq.unsqueeze(0)
        with torch.no_grad():
            lstm_out, _ = lstm(seq)

            att_score = relu(attn(lstm_out))
            norm_score = torch.softmax(att_score, dim=1)
            cntx = (norm_score * lstm_out).sum(dim=1)

            video_logit = classifier(cntx)
            video_label = torch.tensor([[label]], dtype=torch.float, device=device)
            
            loss = criterion(video_logit, video_label)
            loss = loss.mean()
            prob = torch.sigmoid(video_logit).item()

        loss_item = loss.item()
        running_loss += loss_item
        video_run_loss += loss_item

        losses.append(loss_item)
        labels.append(int(label))
        probabs.append(prob)
        correct += int((prob >= 0.5) == int(label))
        total_videos += 1

        videos_processed += 1

        video_bar.update(1)

avg_loss = running_loss / videos_processed if videos_processed > 0 else 0

end_time = time.time()
vali_time = end_time - start_time
accuracy = correct / total_videos if total_videos > 0 else 0
val_loss = sum(losses) / total_videos if total_videos > 0 else 0
vali_inf_time = vali_time / max(1, total_images)
vali_fps = total_images / vali_time if vali_time > 0 else 0


# print results

print("\n\n\n\n")
print(f"Validation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Log Loss: {val_loss:.4f}")

print(f"Validation Time: {vali_time/60:.2f} minutes")
print(f"Validation Single Inference Time: {vali_inf_time:.2f} seconds")
print(f"Validation FPS: {vali_fps:.2f} images/second")


cm = confusion_matrix([int(x) for x in labels], [1 if p > 0.5 else 0 for p in probabs])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted Real", "Predicted Fake"],
            yticklabels=["Actual Real", "Actual Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (TF, FF, TR, FR)")

plt.text(0.5, 0.2, "TR", ha="center", va="center", color="black", fontsize=12)
plt.text(1.5, 0.2, "FF", ha="center", va="center", color="black", fontsize=12)
plt.text(0.5, 1.2, "FR", ha="center", va="center", color="black", fontsize=12)
plt.text(1.5, 1.2, "TF", ha="center", va="center", color="black", fontsize=12)

# calculate confident errors

CFR=0
CFF=0

for i in range(len(labels)):
    if labels[i]==1 :
        if probabs[i] < 0.3:
            CFR+=1
    else:
        if probabs[i] > 0.7:
            CFF+=1

print(f"Confident False Real: {CFR}")
print(f"Confident False Fake: {CFF}")


from thop import profile
input_tensor = torch.randn(1, 3, 224, 224).to(device)
tmacs, tparams = profile(model, inputs=(input_tensor,))
params_millions = tparams / 1e6
gflops = 2 * tmacs / 1e9
print(f"Params: {params_millions:.2f}M, GFLOPs: {gflops:.2f}")

results_payload = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "model_path": model_path,
    "data_path": data_path,
    "metrics": {
        "vali_time_sec": vali_time,
        "accuracy": accuracy,
        "val_loss": val_loss,
        "vali_inf_time_sec": vali_inf_time,
        "vali_fps": vali_fps
    },
    "confusion_matrix": cm.tolist(),
    "confident_errors": {
        "CFR": CFR,
        "CFF": CFF
    },
    "model_complexity": {
        "params_millions": params_millions,
        "gflops": gflops
    }
}

with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(results_payload, f, indent=2)

print(f"Validation metrics written to {RESULTS_PATH}")
