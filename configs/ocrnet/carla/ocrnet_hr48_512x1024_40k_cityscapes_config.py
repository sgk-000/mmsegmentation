_base_ = '/home/aad13694zb/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py'


# Since we use ony one GPU, BN is used instead of SyncBN
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

dataset_type = 'CarlaCityScapesDataset'
data_root = "/home/aad13694zb/carla-semantic-segmentation/working"

work_dir="/home/aad13694zb/carla-semantic-segmentation/working/outputs/ocrnet_hrnet/40k/lr_0.05/power_0.9"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img",
        ann_dir="labels",
        split='splits/train.txt',
        # pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img",
        ann_dir="labels",
        split='splits/val.txt',
        # pipeline=train_pipeline
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="img",
        ann_dir="labels",
        split='splits/val.txt',
        # pipeline=train_pipeline
        ))

# model = dict(
#     decode_head=[
#     dict(
#         type='FCNHead',
#         in_channels=[48, 96, 192, 384],
#         channels=sum([48, 96, 192, 384]),
#         num_classes=15
#     ),
#     dict(
#         type='OCRHead',
#         in_channels=[48, 96, 192, 384],
#         channels=sum([48, 96, 192, 384]),
#         num_classes=15
#     )]
# )

checkpoint_config = dict(
    interval=1000
)

optimizer = dict(
    lr=0.05
)

lr_config = dict(
    power=0.9
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
    ])

# Let's have a look at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')
