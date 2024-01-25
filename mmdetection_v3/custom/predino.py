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


default_hooks = dict(checkpoint=dict(type='CheckpointHook',interval=1,save_best='auto',max_keep_ckpts=3))
custom_hooks = [dict(type='SubmissionHook')]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=300))

img_scales = [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
             (736, 1333), (768, 1333), (800, 1333)]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=s, keep_ratio=True)
                for s in img_scales
            ],
            [
                # ``RandomFlip`` must be placed before ``Pad``, otherwise
                # bounding box coordinates after flipping cannot be
                # recovered correctly.
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [dict(type='LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'flip', 'flip_direction'))
            ]
        ])
]

# amp
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale = 'dynamic')