import os
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def main():
    annotation = '/data/ephemeral/home/dataset/train.json'

    with open(annotation) as f: 
        data = json.load(f)

    var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']), 1))
    y = np.array([v[1] for v in var])
    groups = np.array([v[0] for v in var])

    img_dir = '/data/ephemeral/home/dataset/yolo/images'

    save_dir = '/data/ephemeral/home/dataset/yolo/split'
    if not os.path.isdir('/data/ephemeral/home/dataset/yolo'):
        os.mkdir('/data/ephemeral/home/dataset/yolo')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    SEED = 42
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        save_train_path = os.path.join(save_dir, f'train_{SEED}_fold_{fold}.txt')
        save_val_path = os.path.join(save_dir, f'val_{SEED}_fold_{fold}.txt')

        train_paths, val_paths = '', ''

        train_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
        val_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])

        for idx in train_img_ids:
            train_paths += os.path.join(img_dir, data['images'][idx]['file_name'][6:]) + '\n'
        for idx in val_img_ids:
            val_paths += os.path.join(img_dir, data['images'][idx]['file_name'][6:]) + '\n'

        with open(save_train_path, 'w') as f:
            f.write(train_paths)
        
        with open(save_val_path, 'w') as f:
            f.write(val_paths)

if __name__ == '__main__':
    main()