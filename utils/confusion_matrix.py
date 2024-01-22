import numpy as np
import argparse
from pycocotools.coco import COCO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

def box_iou_calc(boxes1, boxes2):
    '''
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.Arguments:boxes1 (Array[N, 4])boxes2 (Array[M, 4])
    Returns:iou (Array[N, M]): the NxM matrix containing the pairwiseIoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    '''

    def box_area(box):
      return (box[2] - box[0]) * (box[3] - box[1])
   
    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2]) # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)

    return inter / (area1[:, None] + area2 - inter) # iou = inter / (area1 + area2 - inter)

class ConfusionMatrix:
    def __init__(self, num_classes: int, conf_thr=0.05, iou_thr=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr

    def plot(self, save_path, file_name,
             names=['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']):
        try:
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1e-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={'size': 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background(FP)'] if labels else 'auto',
                       yticklabels=names + ['background(FN)'] if labels else 'auto').set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')

            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            fig.savefig(os.path.join(save_path, file_name), dpi=250)
        except Exception as e:
            print(e)
            pass

    def process_batch(self, detections, labels: np.ndarray):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''

        gt_classes = labels[:, 0].astype(np.int16)
        try:
            detections = detections[detections[:, 4] > self.conf_thr]
        except IndexError or TypeError: 
            for i in range(len(labels)):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)
        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.iou_thr)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]
        all_matches = np.array(all_matches)

        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i in range(len(labels)):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i in range(len(detections)):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

def main(args):
    conf_mat = ConfusionMatrix(num_classes=10, conf_thr=0.05, iou_thr=0.5)
    gt_path = args.gt_json
    pred_path = args.pred_csv

    pred_df = pd.read_csv(pred_path)
    new_pred = []
    gt = []

    file_names = pred_df['image_id'].values.tolist()
    bboxes = pred_df['PredictionString'].values.tolist()

    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            print(f'{file_names[i]} empty box')

    for bbox in bboxes:
        new_pred.append([])
        boxes = np.array(str(bbox).split(' '))

        if len(boxes) % 6 == 1: # last space
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')
        
        for box in boxes: # min_x, min_y, max_x, max_y, conf_score, class
            new_pred[-1].append([float(box[2]), float(box[3]), float(box[4]), float(box[5]), float(box[1]), float(box[0])])

    coco = COCO(gt_path)
    for image_id in sorted(coco.getImgIds()):
        gt.append([])
        ann_ids = coco.getAnnIds(image_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            gt[-1].append([ # class, min_x, min_y, max_x, max_y
                float(ann['category_id']),
                float(ann['bbox'][0]),
                float(ann['bbox'][1]),
                float(ann['bbox'][0]) + float(ann['bbox'][2]),
                float(ann['bbox'][1]) + float(ann['bbox'][3]),
            ])

    for p, g in zip(new_pred, gt):
        conf_mat.process_batch(np.array(p), np.array(g))

    conf_mat.plot(args.save_path, args.file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gt-json', type=str, required=True)
    parser.add_argument('--pred-csv', type=str, required=True)
    parser.add_argument('--save-path', type=str, default='/data/ephemeral/home/level2-objectdetection-cv-03/results_analysis',)
    parser.add_argument('--file-name', type=str, required=True)
    args = parser.parse_args()
    main(args)