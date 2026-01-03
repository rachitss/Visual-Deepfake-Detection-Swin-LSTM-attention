import os, json, glob
import pandas as pd
from tqdm.auto import tqdm
import shutil
import subprocess
import random
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT_DATASET = r"<dataset-path>"
OUTPUT_DIR = r"<output-path>"
FFMPEG_PATH = r"absolute-path-to-ffmpeg.exe"
COMPLETED_FILE = 'completed_compressed_folders.json'

VIDEO_EXT = "*.mp4"
CRF_CHOICES = [ # type of compression and corresponding CRF value
    ("raw", None),
    ("c23", 23),
    ("c30", 30),
    ("c35", 35),
 ]
MAX_RETRIES = 3
MAX_WORKERS = 16
batch_size = 32

# function to run ffmpeg compression
def run_ffmpeg(src_path: str, dst_path: str, crf: Optional[int]) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if crf is None:
        shutil.copy2(src_path, dst_path)
        return
    
    #  build ffmpeg command
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "faster",
        "-c:a", 
        "copy",
        "-movflags",
        "+faststart",
        dst_path,
    ]

    # run ffmpeg command
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            subprocess.run(cmd, check=True)
            break
        except subprocess.CalledProcessError as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"ffmpeg failed for {src_path} (crf={crf})") from exc



def select_crf() -> tuple[str, Optional[int]]: # function to randomly select a CRF value
    return random.choice(CRF_CHOICES)

# load completed directories if file exists, otherwise create it
if os.path.exists(COMPLETED_FILE):
    with open(COMPLETED_FILE, 'r') as f:
        completed_folders = set(json.load(f))
else:
    completed_folders = set()
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(list(completed_folders), f)

leaf_dirs = glob.glob(os.path.join(ROOT_DATASET, '*', '*'))
leaf_dirs = [d for d in leaf_dirs if os.path.isdir(d)]
print(leaf_dirs)

# filter out already completed folders
pending_dirs = [d for d in leaf_dirs if d not in completed_folders]
batch = pending_dirs[:batch_size]

# process each pending directory
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

    videos = sorted(glob.glob(os.path.join(leaf, VIDEO_EXT)))
    compression_records = []

    # process videos with multithreading
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_video = {}

        def process_video(v_src, v_dst): # helper function to process a single video
            label, crf_value = select_crf()
            run_ffmpeg(v_src, v_dst, crf_value)
            return label

        # run video compression tasks
        for vid in videos:
            vid_name = os.path.splitext(os.path.basename(vid))[0]
            target_path = os.path.join(out_folder, f"{vid_name}.mp4")
            future = executor.submit(process_video, vid, target_path)
            future_to_video[future] = vid_name

        # collect results
        for future in as_completed(future_to_video):
            vid_name = future_to_video[future]
            label = future.result()
            compression_records.append({"video_name": f"{vid_name}.mp4", "compression": label})

    # save compression data
    if compression_records:
        log_df = pd.DataFrame(compression_records)
        log_path = os.path.join(out_folder, 'compression_map.csv')
        log_df.to_csv(log_path, index=False)

    # update completed folders file
    completed_folders.add(leaf)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump(list(completed_folders), f)

    print('Folder: ', leaf, ' completed')
