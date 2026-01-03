import pandas as pd
import os
from tqdm.auto import tqdm
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess(dir_path):

    # load metadata
    train_dir = dir_path
    train_df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))
    train_df['path'] = train_df['filename'].apply(lambda x: os.path.join(train_dir, x.split('.')[0]))

    # remove empty folders
    train_df = train_df[train_df['path'].map(lambda x: os.path.exists(x))]

    # create valid dataframe
    valid_train_df = pd.DataFrame(columns=['filename', 'label', 'split', 'original', 'path'])

    for row_idx in tqdm(train_df.index):
        row = train_df.loc[row_idx]
        img_dir = row['path']
        face_paths = glob.glob(f'{img_dir}/*.png')

        # satisfy the minimum number of faces
        if len(face_paths) >= 5:
            face_indices = [
                path.split('\\')[-1].split('.')[0].split('_')[0]
                for path in face_paths
            ]
            max_idx = np.max(np.array(face_indices, dtype=np.uint32))

            selected_paths = []

            # sample 5 faces evenly distributed
            for i in range(5):
                stride = int((max_idx + 1)/(5**2))
                sample = np.linspace(i*stride, max_idx + i*stride, 5).astype(int)

                for idx in sample:
                    paths = glob.glob(f'{img_dir}/{idx}*.png')
                    selected_paths.extend(paths)
                    if len(selected_paths) >= 5: # get enough faces
                        break

                if len(selected_paths) >= 5:  # get enough faces
                    valid_train_df = pd.concat([valid_train_df, pd.DataFrame([row])], ignore_index=True)
                    break

    # encode labels
    valid_train_df['label']=valid_train_df['label'].replace({'FAKE': 1, 'REAL': 0})
    print(valid_train_df.head())

    # print label distribution
    label_count = valid_train_df.groupby('label').count()['filename']
    print(label_count)

    # train test split
    x = valid_train_df['path'].to_numpy()
    y = valid_train_df['label'].to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=224, stratify=y)

    # save splits
    np.savez('train_test_split.npz',
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val)
    

if __name__ == '__main__':
    dir_path = "<path-to-extracted-faces>"
    preprocess(dir_path)
