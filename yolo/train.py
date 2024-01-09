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
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch', '--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', '--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--iou', type=float, default=0.5)

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
