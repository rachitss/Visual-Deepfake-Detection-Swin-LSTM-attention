import os, json, cv2, glob
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm.auto import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor


ROOT_DATASET = r"<dataset-path>"
OUTPUT_DIR = r"<output-path>"
COMPLETED_FILE = 'mtcnn_completed_folders.json'

SCALE = 0.25 # downsample factor
N_FRAMES = 32 # number of frames to extract
IMAGE_BATCH_SIZE = 32
MAX_WORKERS = 16
batch_size = 32

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

class FaceExtractor: # helper function for face extraction
    def __init__(self, detector, n_frames=None, resize=None, batch_size=1, fixed_size=(224, 224)):
        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
        self.batch_size = batch_size
        self.fixed_size = fixed_size

    def __call__(self, video_path, save_dir):

        # extract frames from video
        v_cap = cv2.VideoCapture(video_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # determine frame indices to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        frames = []
        frame_indices = []

        for j in range(v_len):

            # get frames
            success = v_cap.grab()
            if j in sample:
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = frame.resize(self.fixed_size)
                frames.append(frame)
                frame_indices.append(j)

                # extract faces in frame batch
                if len(frames) == self.batch_size:
                    os.makedirs(save_dir, exist_ok=True)
                    save_paths = [os.path.join(save_dir, f'{idx}.png') for idx in frame_indices]
                    self.detector(frames, save_path=save_paths)
                    frames = []
                    frame_indices = []

        # extract remaining frames (last batch)
        if frames:
            os.makedirs(save_dir, exist_ok=True)
            save_paths = [os.path.join(save_dir, f'{idx}.png') for idx in frame_indices]
            self.detector(frames, save_path=save_paths)

        v_cap.release()

face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=DEVICE).eval() # load MTCNN
extractor = FaceExtractor(face_detector, n_frames=N_FRAMES, resize=SCALE, batch_size=IMAGE_BATCH_SIZE) # create extractor object


# load completed directories if file exists, otherwise create it
if os.path.exists(COMPLETED_FILE):
    with open(COMPLETED_FILE, 'r') as f:
        completed_folders = set(json.load(f))
else:
    completed_folders = set()
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(list(completed_folders), f)

# get leaf directories
leaf_dirs = glob.glob(os.path.join(ROOT_DATASET, "*", "*"))
leaf_dirs = [d for d in leaf_dirs if os.path.isdir(d)]
print(leaf_dirs)

# filter out completed directories
pending_dirs = [d for d in leaf_dirs if d not in completed_folders]
batch = pending_dirs[:batch_size]

with torch.no_grad():
    for leaf in tqdm(batch, desc='Leaf folders' ):

        # make output directory
        rel_path   = os.path.relpath(leaf, ROOT_DATASET)
        out_folder = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(out_folder, exist_ok=True)

        # copy metadata if it exists and not already copied
        meta_src = os.path.join(leaf, 'metadata.json')
        meta_dst = os.path.join(out_folder, 'metadata.json')
        if os.path.isfile(meta_src) and not os.path.isfile(meta_dst):
            shutil.copy2(meta_src, meta_dst)

        # # uncomment to copy compression data
        # # copy compression data if it exists and not already copied
        # comp_src = os.path.join(leaf, 'compression_map.csv')
        # comp_dst = os.path.join(out_folder, 'compression_map.csv')
        # if os.path.isfile(comp_src) and not os.path.isfile(comp_dst):
        #     shutil.copy2(comp_src, comp_dst)

        # get videos in the leaf directory
        videos = glob.glob(os.path.join(leaf, '*.mp4'))

        def process_video(video_path): # helper function to process a single video
            vid_name = os.path.splitext(os.path.basename(video_path))[0]
            save_dir = os.path.join(out_folder, vid_name)
            extractor(video_path, save_dir)

        # process videos with multithreading
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            list(tqdm(
                executor.map(process_video, videos),
                total=len(videos),
                desc='Videos',
                leave=False,
                miniters=2,
                dynamic_ncols=True
            ))

        # update completed folders file
        completed_folders.add(leaf)
        with open(COMPLETED_FILE, 'w') as f:
            json.dump(list(completed_folders), f)
        
        print('Folder: ', leaf, ' completed')
