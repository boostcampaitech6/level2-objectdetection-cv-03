import os
import json

def main():
    # 저장할 경로 입력
    save_path = '/data/ephemeral/home/dataset/yolo/labels'
    if not os.path.isdir('/data/ephemeral/home/dataset/yolo'):
        os.mkdir('/data/ephemeral/home/dataset/yolo')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # 읽어올 annotation 경로 입력
    json_path = '/data/ephemeral/home/dataset/train.json'
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    print('start')

    image_info_dict = dict()
    for image_info in json_data['images']:
        image_id = image_info['id']
        image_info_dict[image_id] = image_info

    yolo_info_dict = dict()
    for anno_info in json_data['annotations']:
        image_id = anno_info['image_id']
        image_info = image_info_dict[image_id]
        image_ww, image_hh = image_info['width'], image_info['height']
        file_name = os.path.splitext(os.path.basename(image_info['file_name']))[0]

        cate_id = anno_info['category_id']
        xmin, ymin, ww, hh = anno_info['bbox']
        cx, cy = xmin + (ww / 2), ymin + (hh / 2)
        cx, cy, ww, hh = cx / image_ww, cy / image_hh, ww / image_ww, hh / image_hh

        if file_name not in yolo_info_dict:
            yolo_info_dict[file_name] = f'{cate_id} {cx} {cy} {ww} {hh}\n'
        else:
            yolo_info_dict[file_name] += f'{cate_id} {cx} {cy} {ww} {hh}\n'

    for k, v in yolo_info_dict.items():
        label_path = os.path.join(save_path, k + '.txt')
        with open(label_path, 'w') as f:
            f.write(v)

    print('end')

if __name__ == '__main__':
    main()