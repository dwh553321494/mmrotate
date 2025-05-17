# dataset settings
dataset_type = 'mmdet.CocoDataset'
data_root = '/home/haozi/code/dl/ObjectDetection/mmrotate/data/GZBY_X'
backend_args = None

img_scale = (640, 640)  # width, height

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomChoiceRotate', angles=[90, 180, 270], prob=0.7),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    # avoid bboxes being resized
    dict(
        type='mmdet.LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(type='ConvertMask2BoxType', box_type='qbox'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# metainfo = dict(
#     classes=('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
#              'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#              'basketball-court', 'storage-tank', 'soccer-ball-field',
#              'roundabout', 'harbor', 'swimming-pool', 'helicopter'))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        data_root=data_root,
        ann_file='train10/train.json',
        # ann_file='train/annfiles/',
        data_prefix=dict(img='train10/image/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=5,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        # metainfo=metainfo,
        data_root=data_root,
        ann_file='val435/val.json',
        data_prefix=dict(img='val435/image/'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='RotatedCocoMetric',
    metric='bbox',
    classwise=True,
    backend_args=backend_args)

test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         ann_file='test/test.json',
#         data_prefix=dict(img='test/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/dota/Task1')
