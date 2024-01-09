import os
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TrainDataset

def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))

def train_fn(num_epochs, train_data_loader, optimizer, model, device):
    best_loss = 1000
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = '/data/ephemeral/home/level2-objectdetection-cv-03/faster_rcnn/checkpoints/faster_rcnn_torchvision_checkpoints.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value

def main(config=None):
    cfg = {
        'epochs' : 12,
        'batch_size' : 16,
        'optimizer' : 'SGD',
        'lr' : 0.005
    }
    if config is not None:
        cfg = config
    
    # 데이터셋 불러오기
    annotation = '/data/ephemeral/home/dataset/train.json' # annotation 경로
    data_dir = '/data/ephemeral/home/dataset' # data_dir 경로
    train_dataset = TrainDataset(annotation, data_dir, get_train_transform()) 
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=multiprocessing.cpu_count()//2,
        collate_fn=collate_fn
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    # torchvision model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11 # class 개수= 10 + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    if cfg['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=0.9, weight_decay=0.0005)
    elif cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(params, lr=cfg['lr'])

    num_epochs = cfg['epochs']
    
    # training
    train_fn(num_epochs, train_data_loader, optimizer, model, device)

if __name__ == '__main__':
    main()