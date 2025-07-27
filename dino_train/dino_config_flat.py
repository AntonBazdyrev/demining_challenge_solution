auto_scale_lr = dict(base_batch_size=16)
backend_args = None
base_lr = 2e-05
data_root = '/mnt/data/abazdyrev/aijam/train_coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=10, save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 5
metainfo = dict(
    classes=(
        'Explosives',
        'Anti-personnel mine',
        'Anti-vehicle mine',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            20,
            220,
            60,
        ),
        (
            20,
            20,
            220,
        ),
    ])
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=3,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    num_feature_levels=5,
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
num_levels = 5
optim_wrapper = dict(
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=10, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=2,
        begin=2,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        eta_min=1.0000000000000002e-06,
        type='CosineAnnealingLR'),
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
resume = False
test_cfg = None
test_dataloader = None
test_evaluator = None
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_batch_size_per_gpu = 2
train_cfg = dict(max_epochs=5, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='train/images/'),
        data_root='/mnt/data/abazdyrev/aijam/train_coco/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'Explosives',
                'Anti-personnel mine',
                'Anti-vehicle mine',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    20,
                    220,
                    60,
                ),
                (
                    20,
                    20,
                    220,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                allow_negative_crop=False,
                crop_size=(
                    640,
                    640,
                ),
                type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    640,
                    640,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
vis_backends = [
    dict(type='LocalVisBackend'),
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        metainfo=metainfo,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs')
        ]
    )
)
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
workflow = [
    (
        'train',
        1,
    ),
]
