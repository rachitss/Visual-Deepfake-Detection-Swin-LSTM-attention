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


def train():

    loss_path = "<path-to-save-train-loss>"
    model_path = '<path-to-save-model>'
    data_path = '<path-to-train-test-split>'
    BATCH_SIZE = 32
    MAX_IO_WORKERS = 16
    PREFETCH_WORKERS = 16

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device: {device}')

    # load data
    data = np.load(data_path, allow_pickle=True)
    x_train = data['x_train']
    y_train = data['y_train']

    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", use_fast=True) # image processor object
    model = AutoModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224") # model object
    hidden_size = model.config.hidden_size  # feature size for LSTM input
    
    # LSTM object
    lstm = nn.LSTM(
        input_size=hidden_size, # from transformer hidden size
        hidden_size=256,
        num_layers=1,
        batch_first=True,
        bidirectional=False)

    attn = nn.Linear(256,1) # attention layer
    relu = nn.ReLU() # ReLU activation function
    classifier = nn.Linear(256, 1) # final classification layer
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # balanced Binary Cross Entropy Loss

    # Moving model to device and setting to train
    model = model.to(device)
    model.train()
    relu = relu.to(device)
    relu.train()
    attn = attn.to(device)
    attn.train()
    lstm = lstm.to(device)
    lstm.train()
    classifier = classifier.to(device)
    classifier.train()

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    print(len(model.encoder.layers),"total layers")

    # unfreeze n layer parameters
    n = 1
    for layer in model.encoder.layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True
        print(f'Unfreezing {n} layers')

    for param in classifier.parameters():
        param.requires_grad = True

    for param in lstm.parameters():
        param.requires_grad = True

    for param in attn.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad == True], 'lr': 1e-4},
        {'params': classifier.parameters(), 'lr': 1e-3},
        {'params': lstm.parameters(), 'lr': 1e-3},
        {'params': attn.parameters(), 'lr': 1e-3},
        ])

    # training Loop
    if os.path.exists(model_path):
        print("Loading existing model weights...")
        state = torch.load(model_path, map_location='cpu')        

        # load model weights
        model.load_state_dict(state['backbone'], strict=False)
        lstm.load_state_dict(state['lstm'])
        classifier.load_state_dict(state['classifier'])
        optimizer.load_state_dict(state['optimizer'])
        attn.load_state_dict(state['attn'])

        model = model.to(device)
        lstm = lstm.to(device)
        classifier = classifier.to(device)
        attn = attn.to(device)

        # load training loss
        t_l = np.load(loss_path)

        print(f"Last Epoch Training Log Loss: {t_l[-1]:.4f}")
        print(f"Number of epochs trained: {len(t_l)}")

        print("Resuming training with loaded model weights.")
    else:
        print("Model weights not found, starting training...")


    print("\n\nEnter Number of epochs to train: ")
    num_epochs = int(input().strip())

    batch_size = BATCH_SIZE
    video_paths = x_train
    video_labels = y_train
    total_batches = 0
    train_loss = np.zeros(1)
    start_time = time.time()

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

    epoch_bar = tqdm(total=num_epochs, desc='Epochs', ncols=100, position=0)
    for epoch in range(num_epochs):
        model.train()
        lstm.train()
        classifier.train()
        attn.train()

        running_loss = 0.0
        total_batches = 0
        videos_processed = 0

        # shuffle data each epoch
        perm = np.random.permutation(len(video_paths))
        shuffled_paths = video_paths[perm]
        shuffled_labels = video_labels[perm]

        video_bar = tqdm(total=len(shuffled_paths), desc='Videos', ncols=100, position=1, leave=False)
        video_iter = iter(zip(shuffled_paths, shuffled_labels))

        # prefetch video batches using ThreadPoolExecutor
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
                    outputs = model(**inputs, output_hidden_states=False, return_dict=True)

                    pooler_output = outputs.pooler_output
                    frame_feats.append(pooler_output)

                    batches += 1
                    total_batches += 1
                
                # process sequence through LSTM, attention layer and classifier
                if frame_feats:
                    seq = torch.cat(frame_feats, dim=0)
                    seq = seq.unsqueeze(0)
                    lstm_out, _ = lstm(seq)

                    att_score = relu(attn(lstm_out))
                    norm_score = torch.softmax(att_score, dim=1)
                    cntx = (norm_score * lstm_out).sum(dim=1)

                    video_logit = classifier(cntx)
                    video_label = torch.tensor([[label]], dtype=torch.float, device=device)

                    # backpropagation
                    optimizer.zero_grad()
                    loss = criterion(video_logit, video_label)
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()

                    loss_item = loss.item()
                    running_loss += loss_item
                    video_run_loss += loss_item

                    videos_processed += 1

                video_bar.update(1)

        avg_loss = running_loss / videos_processed if videos_processed > 0 else 0
        train_loss[0] = avg_loss

        # save model backbone and temporal head
        torch.save({
            'backbone': model.state_dict(),
            'lstm': lstm.state_dict(),
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'attn': attn.state_dict(),
        }, model_path)

        # save training loss
        if os.path.exists(loss_path):
            # loading existing training loss
            training_loss = np.load(loss_path)
            training_loss = np.concatenate((training_loss, train_loss))
            np.save(loss_path, training_loss)

        else:
            training_loss = train_loss
            np.save(loss_path, training_loss)

        epoch_bar.update(1)
        epoch_bar.set_postfix(Curr_Loss=f'{avg_loss:.4f}')

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training completed in {training_time/60:.2f} minutes")

if __name__ == '__main__':
    train()
