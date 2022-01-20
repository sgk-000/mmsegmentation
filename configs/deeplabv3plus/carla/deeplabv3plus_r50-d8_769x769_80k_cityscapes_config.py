_base_ = '/home/aad13694zb/carla-semantic-segmentation/ocrnet/ocrnet_hr48_512x1024_80k_cityscapes.py'


# Since we use ony one GPU, BN is used instead of SyncBN
# cfg.norm_cfg = dict(type='BN', requires_grad=True)
# cfg.model.backbone.norm_cfg = cfg.norm_cfg
# cfg.model.decode_head.norm_cfg = cfg.norm_cfg
# cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

dataset_type = 'CarlaCityScapesDataset'
data_root = "/home/aad13694zb/carla-semantic-segmentation/working"

work_dir="/home/aad13694zb/carla-semantic-segmentation/working/outputs"

data = dict(
    samples_per_gpu = 3,
    workers_per_gpu=3,
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
    auxiliary_head=dict(
        num_classes=15
    )
)

checkpoint_config = dict(
    interval=1000
)

# cfg.checkpoint_config.interval = 1000

# cfg.model.decode_head.num_classes = 15
# cfg.model.auxiliary_head.num_classes = 15

# Modify dataset type and path

# cfg.data.samples_per_gpu = 4
# cfg.data.workers_per_gpu=4


# cfg.data.train.type = cfg.dataset_type
# cfg.data.train.data_root = cfg.data_root
# cfg.data.train.img_dir = "img"
# cfg.data.train.ann_dir = "labels"
# cfg.data.train.pipeline = cfg.train_pipeline
# cfg.data.train.split = 'splits/train.txt'

# cfg.data.val.type = cfg.dataset_type
# cfg.data.val.data_root = cfg.data_root
# cfg.data.val.img_dir = "img"
# cfg.data.val.ann_dir = "labels"
# cfg.data.val.pipeline = cfg.test_pipeline
# cfg.data.val.split = 'splits/val.txt'

# cfg.data.test.type = cfg.dataset_type
# cfg.data.test.data_root = cfg.data_root
# cfg.data.test.img_dir = "img"
# cfg.data.test.ann_dir = "labels"
# cfg.data.test.pipeline = cfg.test_pipeline
# cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
#cfg.load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug/deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth'

# Set up working dir to save files and logs.
# cfg.work_dir = './outputs/'

#cfg.runner.max_iters = 200
#cfg.log_config.interval = 10
#cfg.evaluation.interval = 200
# cfg.checkpoint_config.interval = 1000

# cfg.dist_params = dict(backend='nccl')

# Set seed to facitate reproducing the result
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.gpu_ids = range(4)

# Let's have a look at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')
