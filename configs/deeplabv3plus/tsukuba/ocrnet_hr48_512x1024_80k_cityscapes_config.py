_base_ = '/home/digital/mmsegmentation/configs/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes.py'

dataset_type = 'TsukubaCityScapesDataset'
data_root = "/home/digital/sgk/data/tsukuba/working"

work_dir="/home/digital/sgk/data/tsukuba/working/outputs/ocrnet/40k/lr_0.01/power_0.6"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="imgs",
        ann_dir="labels",
        split='splits/train.txt',
        # pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="imgs",
        ann_dir="labels",
        split='splits/val.txt',
        # pipeline=train_pipeline
        ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="imgs",
        ann_dir="labels",
        split='splits/val.txt',
        # pipeline=train_pipeline
        ))


# model = dict(
#     decode_head=dict(
#         num_classes=15
#     ),
#     auxiliary_head = dict(
#         num_classes=15
#     )
# )

checkpoint_config = dict(
    interval=1000
)

optimizer = dict(
    lr=0.01
)

runner = dict(type='IterBasedRunner', max_iters=80000)

lr_config = dict(
    power=0.6
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
    ])

# Let's have a look at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')
