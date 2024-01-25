import os
import copy
import multiprocessing
import torch
from tqdm import tqdm
import pandas as pd
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader



# mapper - input data를 어떤 형식으로 return할지
def MyMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

if __name__ == '__main__':
    # Register Dataset
    try:
        register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/dataset/test.json', '/data/ephemeral/home/dataset/')
    except AssertionError:
        pass

    # config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))

    # config 수정하기
    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WOREKRS = multiprocessing.cpu_count()//2

    cfg.OUTPUT_DIR = '/data/ephemeral/home/level2-objectdetection-cv-03/detectron2/output'

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_0002999.pth')

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3

    # model
    predictor = DefaultPredictor(cfg)

    # test loader
    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', MyMapper)

    # output 뽑은 후 sumbmission 양식에 맞게 후처리 
    prediction_strings = []
    file_names = []

    class_num = 10

    for data in tqdm(test_loader):
        
        prediction_string = ''
        
        data = data[0]
        
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace('/data/ephemeral/home/dataset/',''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission_det2.csv'), index=None)