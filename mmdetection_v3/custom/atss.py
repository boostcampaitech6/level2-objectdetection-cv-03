_base_ = [
    '../configs/_base_/schedules/schedule_1x.py', 
    '../configs/_base_/default_runtime.py'
]

# model
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_20220509_100315-bc5b6516.pth'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=128),
    backbone=dict(
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
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=[
        dict(
            type='FPN',
            in_channels=[384, 768, 1536],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            # disable zero_init_offset to follow official implementation
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=10,
        in_channels=256,
        pred_kernel_size=1,  # follow DyHead official implementation
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),  # follow DyHead official implementation
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None
)

# amp
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale = 'dynamic')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[2, 5, 8, 11],
        gamma=0.5)
]

# wandb logger
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'newmmdetection',
            'entity': 'funfunfun',
            'name': 'Final2_atss_lr_fold4'
         })
]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# logger val
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3))

workflow = [('train', 1), ('val', 1)]

# custom_hooks = [
#     dict(type='SubmissionHook')
# ]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=300))
img_scales=[(2000, 480), (2000, 1200)]
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

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(2000, 480), (2000, 1200)],
        keep_ratio=True,
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True, backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2000, 1200), keep_ratio=True, backend='pillow'),
    # dict(
    #     type='TestTimeAug',
    #     transforms=[
    #         dict(type='Resize', scale=(1333, 1280), keep_ratio=True),
    #     ]
    # ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
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
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=2,
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
        pipeline=val_pipeline,
        metainfo=dict(classes=classes),
        backend_args=backend_args
    )
)
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
        backend_args=backend_args
    )
)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'split_convert/val_42_fold_4.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args
)
