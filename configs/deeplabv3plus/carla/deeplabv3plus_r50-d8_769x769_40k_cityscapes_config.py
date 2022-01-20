_base_ = '/home/aad13694zb/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_40k_cityscapes.py'

dataset_type = 'CarlaCityScapesDataset'
data_root = "/home/aad13694zb/carla-semantic-segmentation/working"

work_dir="/home/aad13694zb/carla-semantic-segmentation/working/outputs/deeplabv3plus/40k/lr_0.005/power_0.9"

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


model = dict(
    decode_head=dict(
        num_classes=15
    ),
    auxiliary_head = dict(
        num_classes=15
    )
)

checkpoint_config = dict(
    interval=8000
)

optimizer = dict(
    lr=0.005
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
