# Visual Deepfake Detection with Swin-Tiny + LSTM Attention

A complete deepfake detection workflow built around a hybrid architecture that merges the Swin-Tiny transformer for spatial feature extraction with an LSTM + attention head for temporal sequences. The project trains on the full DeepFake Detection Challenge (DFDC) dataset, then serves the resulting model through a FastAPI + Streamlit web application.

## Highlights
- **Novel architecture**: $\text{SwinTiny} \rightarrow \text{LSTM} \rightarrow \text{Attention} \rightarrow \text{Classifier}$ keeps global spatial context while summarizing long-range temporal dynamics.
- **End-to-end tooling**: automated face extraction, dataset curation, balancing, compression, training, evaluation, visualization, and serving.
- **DFDC-scale readiness**: every script is designed to operate on the complete DFDC corpus with resumable processing and batched execution.

## Dataset
- **Source**: Full DFDC videos (train splits and metadata).
- **Preparation workflow**:
  1. Copy raw DFDC folders locally.
  2. Configure `<dataset-path>` and `<output-path>` placeholders inside the scripts to point to your storage layout.
  3. Use the provided utilities to extract faces, stratify splits, balance classes, and optionally generate compressed variants for stress-testing.

## Repository Layout
```
Training-and-Evaluation/
Web-App/
```
Each folder is self-contained and can be used independently (e.g., training on a workstation, serving from a containerized API).

## Training and Evaluation Pipeline
Follow the scripts in this order for a smooth DFDC run:

1. **Face extraction** – [Training-and-Evaluation/extract_faces_MTCNN.py](Training-and-Evaluation/extract_faces_MTCNN.py)
   - Uses `facenet-pytorch` MTCNN to detect faces across all DFDC videos.
   - Saves uniformly sampled frames (default 32) per video and tracks progress via `mtcnn_completed_folders.json` so you can resume large jobs.

2. **Dataset preprocessing** – [Training-and-Evaluation/preprocess_data.py](Training-and-Evaluation/preprocess_data.py)
   - Consumes DFDC `metadata.csv`, filters clips with ≥5 high-quality faces, encodes labels (`REAL=0`, `FAKE=1`), and writes a stratified `train_test_split.npz`.

3. **Optional dataset checks**
   - Inspect the split stats with [Training-and-Evaluation/check_data.py](Training-and-Evaluation/check_data.py).
   - Undersample to balance real/fake distributions via [Training-and-Evaluation/balance_data.py](Training-and-Evaluation/balance_data.py).

4. **Optional compression sweeps** – [Training-and-Evaluation/compression.py](Training-and-Evaluation/compression.py)
   - Batch-compresses videos with ffmpeg at multiple CRF levels (`raw`, `c23`, `c30`, `c35`) to benchmark robustness. Progress is logged in `completed_compressed_folders.json`.

5. **Model training** – [Training-and-Evaluation/model_training.py](Training-and-Evaluation/model_training.py)
   - Loads Swin-Tiny weights, selectively unfreezes the last encoder block, and stacks an `nn.LSTM` + attention pooling + linear head.
   - Streams face crops per video, encodes batches with Swin, aggregates via LSTM attention, and optimizes BCE loss with class balancing.
   - Saves checkpoints (`model_path`) and loss traces (`loss_path`) after every epoch; training can resume automatically.

6. **Loss visualization** – [Training-and-Evaluation/plot_loss.py](Training-and-Evaluation/plot_loss.py)
   - Quick Matplotlib helper to plot the saved log-loss curve and inspect convergence speed.

7. **Model evaluation** – [Training-and-Evaluation/model_evaluation.py](Training-and-Evaluation/model_evaluation.py)
   - Mirrors the training dataloader but runs in eval mode to obtain accuracy, log-loss, FPS, confident error counts, confusion matrix, and complexity (params + GFLOPs via `thop`).
   - Stores a JSON report at `RESULTS_PATH` for downstream analysis.

8. **Result reporting** – [Training-and-Evaluation/results.py](Training-and-Evaluation/results.py)
   - Pretty-prints the JSON metrics and renders the confusion matrix for presentations or lab notes.

### Running the pipeline
```bash
# 1. Activate your environment
python -m venv .venv && .venv\Scripts\activate
pip install -r Web-App/requirements.txt  # contains all shared deps

# 2. Edit the scripts to set dataset/model paths
# 3. Run each stage sequentially
python Training-and-Evaluation/extract_faces_MTCNN.py
python Training-and-Evaluation/preprocess_data.py
python Training-and-Evaluation/balance_data.py  # optional
python Training-and-Evaluation/model_training.py
python Training-and-Evaluation/model_evaluation.py
python Training-and-Evaluation/results.py
```

## Web Application Stack
Once the model is trained (copy `model.pth` into `Web-App/app/`), you can expose it through the bundled API + UI stack.

### FastAPI microservice
- **Entry point**: [Web-App/app/api_service.py](Web-App/app/api_service.py)
- Loads the Swin + LSTM + attention weights once at startup via [Web-App/app/infer.py](Web-App/app/infer.py).
- `/health` reports the active device; `/inference` streams uploaded videos to `run_inference`, which:
  1. Extracts faces on-the-fly using [Web-App/app/extract_faces.py](Web-App/app/extract_faces.py).
  2. Encodes all frames with Swin-Tiny, pools temporally through the LSTM + attention head, and returns the predicted label plus confidence.
- Launch with [Web-App/app/run_api.py](Web-App/app/run_api.py):
  ```bash
  cd Web-App/app
  uvicorn api_service:app --host 0.0.0.0 --port 8000
  ```

### Streamlit front-end
- **App**: [Web-App/app/webapp.py](Web-App/app/webapp.py)
- Provides a lightweight client that checks API health, uploads `.mp4` files, and visualizes predictions.
- Start locally:
  ```bash
  cd Web-App
  streamlit run app/webapp.py
  ```
- Set `DEEPFAKE_API_URL` to point to your FastAPI instance if it runs on another host.

### Dockerized deployment
- Use the provided [Web-App/Dockerfile](Web-App/Dockerfile) to build a CPU-only image that runs both FastAPI (port 8000) and Streamlit (port 8501):
  ```bash
  cd Web-App
  docker build -t deepfake-detector .
  docker run -p 8000:8000 -p 8501:8501 deepfake-detector
  ```

## Configuration Checklist
- Replace every placeholder such as `<dataset-path>`, `<output-path>`, `<path-to-train-test-split>`, `<path-to-save-model>`, and `<path-to-saved-results>` with valid absolute paths before execution.
- Ensure ffmpeg is installed (system package on Linux, binary on Windows) for both compression and face extraction scripts.
- Keep `model.pth` synchronized between the training outputs and the `Web-App/app/` folder used for inference.



