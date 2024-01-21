

## **Trash Object Detection for Recycling**
## 프로젝트 목표
환경 부담을 줄이기 위한 방법 중 하나는 쓰레기를 효과적으로 분리하는 것입니다. 따라서 본 프로젝트는 사진에서 쓰레기를 감지하는 모델을 만들어, **정확한 쓰레기 분리**를 돕는 것을 목표로 합니다.


## 프로젝트 상세
### Dataset
- Format: COCO format
- Total image : 9,754장(Train set 4,883장)
- Class: 10개 (General trash, Paper, Paper pack, Metal, Class, Plastic, Styrofoam, Plastic bag, Battery, Clothing)
- Image Resolution: 1024\*1024

### Evaluation metric
- mAP50

### Computing Environment
- GPU v100

### Framework
- MMDetection v3

### Cooperation tools
- Slack, Notion
- Github, Wandb


<br>

## Team Introduction

### Members
| 팀원 | 역할 |
| -- | -- |
| 전체 | - EDA, Data Cleansing<br>- Ensemble 실험 및 전략 수립 (NMS, WBF)<br>- TTA 실험 (Dino, ATSS) |
| 강현서 | - MMDetection 라이브러리 실험 <br> - Cleansing Data 비교 실험 (Faster-RCNN) <br> - Learning Rate Scheduler 실험(Dino, ATSS) <br> 
| 김정택 | - Cascade RCNN 모델링 (MMDetection)<br>- Cleansing Data 비교 실험 ( YololV8s )<br>- Data Augmentation 실험 (YoloV8s, CascadeRCNN)<br>- Cls Loss 실험 ( CascadeRCNN ) |
| 박진영 | -  |
| 선경은 | -  |
| 이선우라이언 | - |
| 최현우 | -  |

<br>

## Process

  

| 분류 | 내용 |
| :--: | -- |
|Data|**Stratified Group K-fold** <br> - Class 비율을 유지하며, 중복되지 않는 5개의 fold로 나눔 <br> - Test set과 유사한 Valid set을 찾기 위해 다양한 SEED 시도 <br> - 팀원 모두 같은 SEED를 사용하여 실험 비교가 유의미하도록 하였음 <br> <br>  **Augmentation** <br> - 객체의 특성을 최대한 보존하는 증강 기법 적용 <br> - Rotate, Flip, Resize <br> <br> **Label Correction** <br> - 비슷한 객체를 서로 다르게 라벨링하거나 오라벨링 하는 케이스에 대해 correction 진행<br> - 객관성을 유지하기 위해 팀원들 간 합의를 통해 매뉴얼을 정함
|Model| - 최대한 일정 점수 이상(0.6mAP)의 모델을 다양하게 만들어서 앙상블에 이용하고자 함<br> - MMDetection v3 라이브러리에서 단일 모델 중 SOTA 랭킹이 높았던 모델 우선적 사용<br> - 각 모델에 대해 적합한 lr scheduler를 실험적으로 찾아냄<br> - 사용한 모델 상세는 아래의 Result 섹션 참고바람<br>
|Other Methods|**Ensemble** <br> 아래의 항목들을 고려하며 앙상블 실험을 진행함<br>- 앙상블 방법 (wbf, nms)<br> - IOU <br>- Skip box threshold<br>- 각 모델에 준 Weight<br>- Calibration 여부<br>- 1-Stage Model / 2-Stage Model 

<br>


## Result

### Used Models

| Method | Backbone | mAP50 | mAP50(LB) |
| :--: | :--: | :--: | :--: |
|**YOLOv8l**|-| 0.6104 | 0.6140 |
|**Cascade RCNN**| Swin-L | 0.6134 | 0.6307 |
|**Dino**| Swin-L | 0.7098 | 0.7024 |
|**ATSS**| Swin-L | 0.6986 | 0.7156 |

<br>



### Final Score

🥇 **Private LB : 1st**
<img  src='https://velog.velcdn.com/images/jelly9999/post/7fdcfafa-a4f4-49f5-89e0-663901d45597/image.png'  height=200  width=900px></img>
