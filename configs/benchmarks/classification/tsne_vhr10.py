# _base_ = 'tsne_imagenet.py'

# data = dict(
#     extract=dict(
#         data_source=dict(
#             data_prefix = '/media/ls/disk1/NWPU VHR-10 dataset 3/VOCdevkit/VOC2007/JPEGImages',
#             ann_file = '/media/ls/disk1/NWPU VHR-10 dataset 3/VOCdevkit/VOC2007/ImageSets/Main/val.txt',        
#             )
#     )
# )
dataset_type = 'mmcls.ImageNet'
data_root = '/media/ls/disk1/NWPU VHR-10 dataset 3/imagenet/'# 'data/imagenet/'
file_client_args = dict(backend='disk')
name = 'imagenet_val'

extract_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs'),
]

extract_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='/media/ls/disk1/NWPU VHR-10 dataset 3/imagenet/',
        ann_file='meta/train.txt',
        data_prefix='train/',
        pipeline=extract_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# pooling cfg
pool_cfg = dict(type='MultiPooling', in_indices=(1, 2, 3, 4))
