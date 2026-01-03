import os
import tempfile
from pathlib import Path

import torch
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')



class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in tqdm(range(v_len), desc="Extracting faces"):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()

def get_frames(video_path: str, tmp_root: str = 'MTCNN_EXTRACTED') -> str:
    """Extract faces from the provided video path into a temp directory."""
    os.makedirs(tmp_root, exist_ok=True)

    scale = 0.25
    n_frames = None

    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
    face_extractor = FaceExtractor(detector=face_detector, n_frames=n_frames, resize=scale)

    with torch.no_grad():
        stem = Path(video_path).stem or 'video'
        save_dir = tempfile.mkdtemp(prefix=f'{stem}_', dir=tmp_root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        face_extractor(video_path, save_dir)
        return save_dir

