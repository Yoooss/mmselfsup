_base_ = 'mmdet::pascal_voc/faster-rcnn_r50-caffe-c4_ms-18k_voc0712.py'
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712.py

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm',
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch'),
        bbox_head=dict(num_classes=10)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
dataset_type = 'VOCDataset'
data_root = '/media/ls/disk1/NWPU VHR-10 dataset 3/VOCdevkit/'

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='VOCDataset',
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        ))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline,data_root=data_root,))
test_dataloader = val_dataloader

train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=24, val_interval=4)


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=24000,
#         by_epoch=False,
#         milestones=[18000, 22000],
#         gamma=0.1)
# ]

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(by_epoch=True, interval=4))

# log_processor = dict(by_epoch=True)

custom_imports = dict(
    imports=['mmselfsup.evaluation.functional.res_layer_extra_norm'],
    allow_failed_imports=False)
