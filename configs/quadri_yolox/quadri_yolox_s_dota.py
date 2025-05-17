_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py', '../_base_/datasets/dota_qbox.py',
]


model = dict(
    type='mmdet.YOLOX',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        # batch_augments=[
        #     dict(
        #         type='mmdet.BatchSyncRandomResize',
        #         random_size_range=(480, 800),
        #         size_divisor=32,
        #         interval=10)
        # ]),
    ),
    backbone=dict(
        type='mmdet.CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='mmdet.YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='QuadriYOLOXHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='QuadriLoss',
            # iou_loss_type=dict(
            #     type='mmdet.IoULoss',
            #     mode='square',
            #     eps=1e-16,
            #     reduction='sum',
            #     loss_weight=2.5,
            # ),
            loss_type=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0,
                reduction='mean',
                loss_weight=5,
            ),
            loss_weight=5),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='mmdet.L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5)),
    # train_cfg=dict(assigner=dict(type='QuadriSimOTAAssigner', center_radius=2.5)),

    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms_quadri', iou_threshold=0.6))
)


train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning rate
param_scheduler = [
    # dict(
    #     # use quadratic formula to warm up 5 epochs
    #     # and lr is updated by iteration
    #     # TODO: fix default scope in get function
    #     type='mmdet.QuadraticWarmupLR',
    #     by_epoch=False,
    #     begin=0,
    #     end=20,
    #     ),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=0,
        T_max=1000,
        end=1000,
        by_epoch=False,
        ),
    # dict(
    #     # use fixed lr during last 15 epochs
    #     type='ConstantLR',
    #     by_epoch=False,
    #     factor=1,
    #     begin=900,
    #     end=1000,
    # )
]