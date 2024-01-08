from ultralytics import YOLO
import argparse
import pandas as pd
import os

def main(opt):
    model = YOLO(opt.weight)
    results = model(opt.source)

    df = pd.DataFrame(columns=['PredictionString', 'image_id'])
    prediction_arr = []
    
    for result in results:
        for box in result.boxes:
            c = int(box.cls)  
            confidence = float(box.conf)
            *xyxy, = box.xyxy

            prediction_arr.append(str(c))
            prediction_arr.append(str(confidence))
            for coord in list(map(float, xyxy[0])):
                prediction_arr.append(str(coord))

        paths = result.path.split('/')
        image_id = '/'.join(paths[-2:])
        prediction_str = ' '.join(prediction_arr)

        df = df.append(pd.DataFrame({'PredictionString':[prediction_str], 'image_id':[image_id]}), ignore_index=True)
        prediction_arr = []
        
    save_dir = opt.save_dir
    name = opt.name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    df.to_csv(os.path.join(save_dir, f'{name}.csv'), index=False)


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--source', type=str, default='/data/ephemeral/home/dataset/test')
    parser.add_argument('--save-dir', type=str, default='/data/ephemeral/home/level2-objectdetection-cv-03/yolo/results')
    parser.add_argument('--name', type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
