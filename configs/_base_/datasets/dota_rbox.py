
# dataset settings
dataset_type = 'PVDataset'
data_root = '/home/haozi/code/dl/ObjectDetection/mmrotate/data/GZBY_X'
# data_root = '/mnt/h/dwh_backup/dataset/data_4/'
backend_args = None

img_scale = (320, 320)  # width, height
angle_version = 'oc'
albu_train_transforms = [
    # dict(type='RandomRotate90', p=0.7),
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=0,
    #     interpolation=1,
    #     p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.35),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1),
            dict(type='MedianBlur', blur_limit=3, p=1)
        ],
        p=0.2),
    # dict(type='HorizontalFlip', p=0.5),
    # dict(type='VerticalFlip', p=0.5),
]


train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    # dict(type='RandomChoiceRotate', angles=[90, 180, 270], prob=0.8),
    # dict(
    #     type='mmdet.RandomFlip',
    #     prob=0.75,
    #     direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmdet.Albu',
        transforms = albu_train_transforms,
        # bbox_params=dict(
        #     type='BboxParams',
        #     format='pascal_voc',
        #     label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
        #     min_visibility=0.0,
        #     filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='RegularizeRotatedBox', # 统一旋转框表示形式
         angle_version=angle_version), # 根据角度的定义方式进行
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=5,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train_instance10/txt/',
        # ann_file='train/annfiles/',
        data_prefix=dict(img_path='train_instance10/image/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=5,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test_instance1200/txt/',
        data_prefix=dict(img_path='test_instance1200/image/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DOTAMetric', metric='mAP', 
    iou_thrs=[0.5, 0.75], 
    predict_box_type='rbox', iou_thr = 0.25)
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
#         data_root=data_root,
#         data_prefix=dict(img_path='test/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
test_evaluator = dict(
    iou_thr=0.25,
    iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    metric='mAP',
    merge_patches=False,
    format_only=True,
    output_dir='/home/haozi/code/dl/ObjectDetection/mmrotate/data/GZBY_X/test_instance1200/results/rotated_yolox_s_iter_5000_10',
    predict_box_type='rbox',
    type='DOTAMetric')
