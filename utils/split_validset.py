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

    save_dir = '/data/ephemeral/home/dataset/split'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    SEED = 42
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):

        train_img_ids = set([data['annotations'][idx]['image_id'] for idx in train_idx])
        train_imgs = [data['images'][idx] for idx in train_img_ids]

        train_data = {
            'images': train_imgs,
            'categories': data['categories'],
            'annotations': [data['annotations'][idx] for idx in train_idx]
        }
        
        val_img_ids = set([data['annotations'][idx]['image_id'] for idx in val_idx])
        val_imgs = [data['images'][idx] for idx in val_img_ids]
        
        val_data = {
            'images': val_imgs,
            'categories' : data['categories'],
            'annotations': [data['annotations'][idx] for idx in val_idx]
        }
        
        train_path = os.path.join(save_dir, f'train_{SEED}_fold_{fold}.json')
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=4)
            
        val_path = os.path.join(save_dir, f'val_{SEED}_fold_{fold}.json')
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=4)

if __name__ == '__main__':
    main()