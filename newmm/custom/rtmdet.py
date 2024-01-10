
_base_ = '../configs/rtmdet/rtmdet_x_8xb32-300e_coco.py'


model = dict(bbox_head=dict(num_classes=10))

vis_backends = [
    dict(type='WandbVisBackend',
         init_kwargs={
             'project':'newmmdetection',
             'entity':'ai-tech-6th-funfun',
             
         })
]
visualizer = dict(type='DetLocalVisualizer',vis_backends=vis_backends,name='visualizer')

