from ultralytics import YOLO
import argparse


def main(opt):
    model = YOLO(opt.weight)

    file_lists = []    
    
    batch_size = 100
    file_paths = [opt.vaild_file]

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines), batch_size):
                batch_lines = lines[i:i+batch_size]
                file_lists = []
                for line in batch_lines:
                    jpg_path = line.strip()
                    if jpg_path.endswith('.jpg'):
                        file_lists.append(jpg_path)

                model(file_lists, save=True)

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--vaild-file', type=str, default='/data/ephemeral/home/dataset/yolo/split/val_42_fold_1.txt')

    return parser.parse_args()

    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)