from ultralytics import YOLO
import argparse

def main(opt):
    model = YOLO(opt.model)
    model.train(cfg='./custom.yaml', **vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch', '--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', '--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--name', type=str, default='')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='Adamax')
    parser.add_argument('--lr0', type=float, default=1e-3)

    # loss
    parser.add_argument('--box', type=float, default=7.5)
    parser.add_argument('--cls', type=float, default=1.0)
    parser.add_argument('--dfl', type=float, default=1.5)

    # val
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--iou', type=float, default=0.5)

    # augment
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.4)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
