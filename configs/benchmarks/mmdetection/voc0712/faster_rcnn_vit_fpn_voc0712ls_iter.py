_base_ = 'mmdet::pascal_voc/faster-rcnn_swin-c4_voc0712.py'
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712.py
#custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32)

norm_cfg = dict(type='LN', requires_grad=True) #SyncBN
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.VisionTransformer',
        arch='base',
        patch_size=16,
        out_indices=(0, 1, 2, 3),
        #mask_ratio=0.75,
        # init_cfg=[
        #     dict(type='Xavier', distribution='uniform', layer='Linear'),
        #     dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        # ]
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=10))) # 1124: ls 10; used to be 2 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        # scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
        #         (1333, 608), (1333, 640), (1333, 672), (1333, 704),
        #         (1333, 736), (1333, 768), (1333, 800)],
        scales = [(666, 240), (666, 256), (666,272), (666, 288),
                   (666, 304), (666, 320), (666, 336), (666, 352),
                   (666, 368), (666, 384), (666, 400)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(666, 400), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
dataset_type = 'VOCDataset'
data_root = '/media/ls/disk1/NWPU VHR-10 dataset 3/VOCdevkit/'

train_dataloader = dict(
    batch_size=4,
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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=240000, val_interval=2000)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24000,
        by_epoch=False,
        milestones=[18000, 22000],
        gamma=0.1)
]

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=1))

log_processor = dict(by_epoch=False)
auto_scale_lr = dict(enable=False, base_batch_size=16) #1211添加 可测试加和不加哪个性能更好

custom_imports = dict(
    imports=['mmselfsup.models.utils.res_layer_extra_norm'],
    allow_failed_imports=False)
