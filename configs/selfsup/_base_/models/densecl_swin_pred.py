# model settings
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
#custom_imports = dict(imports=['/home/ls/mmselfsupv1/mmselfsup/models/heads/denseclls_head'], allow_failed_imports=False)
model = dict(
    type='DenseLs',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    loss_lambda=0.5,
    predictor =dict(
        type='NonLinearNeck',
        in_channels=128,
        hid_channels=4096,
        out_channels=128,
        num_layers=2,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=False),
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='mmcls.SwinTransformer',
        arch='T',
        #img_size=192,
        out_indices = (3,),#1024
        #stage_cfgs=dict(block_cfgs=dict(window_size=6))
        ),
    neck=dict(
        type='DenseCLNeck',
        in_channels=768,
        hid_channels=2048,
        out_channels=128,
        num_grid=None),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2),
)
