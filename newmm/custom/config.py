_base_ = '../configs/rtmdet/rtmdet_x_8xb32-300e_coco.py'


model = dict(bbox_head=dict(num_classes=10))