import cv2
import os, glob

def combine_images(combine_list, save_path):
    for data_paths in combine_list:
        if len(data_paths) != len(combine_list[0]):
            print('file count not correct')
            break

    print('start')
    for i in range(len(combine_list[0])):
        img_name = os.path.basename(combine_list[0][i])
        img = cv2.imread(combine_list[0][i])
        for j in range(1, len(combine_list)):
            img = cv2.hconcat([img, cv2.imread(combine_list[j][i])])
        cv2.imwrite(os.path.join(save_path, img_name), img)
    print('end')

gt_42_4_valid_image_paths = sorted(glob.glob('/data/ephemeral/home/dataset/vallid_gt/*'))
yolov5s_42_4_wbf_paths = sorted(glob.glob('/data/ephemeral/home/baseline/yolo/runs/detect/predict/*'))

combine_list = [gt_42_4_valid_image_paths, yolov5s_42_4_wbf_paths]

combine_images(combine_list, '/data/ephemeral/home/baseline/yolo/runs/detect/compare')