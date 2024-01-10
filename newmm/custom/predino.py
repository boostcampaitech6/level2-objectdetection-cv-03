_base_ = '../configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py'

model = dict(bbox_head=dict(num_classes=10))

vis_backends = [
    dict(type='WandbVisBackend',
         init_kwargs={
             'project':'newmmdetection',
             'entity':'ai-tech-6th-funfun',
             
         })
]
visualizer = dict(type='DetLocalVisualizer',vis_backends=vis_backends,name='visualizer')

test_cfg = dict(rcnn=dict(score_thr=0.25))

default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=1,save_best='auto',max_keep_ckpts=3))
custom_hooks = [dict(type='SubmissionHook')]