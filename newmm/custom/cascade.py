_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    '../configs/_base_/default_runtime.py'
]

# model
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
        
    roi_head=dict(
        type='CascadeRoIHead',
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                loss_cls = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9
                    ),
                ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                loss_cls = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9
                    ),
                ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                loss_cls = dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.9
                    ),
                 )
            ],
        ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)))

# wandb logger
vis_backends = [dict(type='WandbVisBackend')]
init_kwargs=dict(project='toy-example')
visualizer = dict(vis_backends=[dict(type='WandbVisBackend', 
                                     init_kwargs=dict(entity='funfunfun',
                                                      project='newmmdetection',
                                                      name='Final_cascade_fold5'))])

# logger val
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3),
    )

# training schedule for 20e
max_epochs = 14
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

# amp
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale = 'dynamic')
# load_from = '/opt/workspace/custom/models/cascade_mask_rcnn_r101_fpn_mstrain_3x_coco_20210628_165236-51a2d363_pretrained.pth'


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[2, 8, 11],
        gamma=0.5)
]

custom_hooks = [dict(type='SubmissionHook')]

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]


# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                            (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                            (1333, 736), (1333, 768), (1333, 800)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(4200, 400), (4200, 500), (4200, 600)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                            (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                            (1333, 736), (1333, 768), (1333, 800)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline= [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    #dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='split_convert/train_42_fold_4.json',
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='split_convert/val_42_fold_4.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=data_root),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'split_convert/val_42_fold_4.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)