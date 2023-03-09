_base_ = ["./ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py"]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        bottleneck_mode="ip",
        pretrained="checkpoints/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth",  # noqa: E251  # noqa: E501
    ),
    cls_head=dict(
        type="I3DHead",
        num_classes=4,  # changed from 400 to 4 for soccer_highlights
        in_channels=2048,
        spatial_type="avg",
        dropout_ratio=0.5,
        init_std=0.01,
    ),
    #Test config
    #Reduce max_testing_views for reduced gpu time
    test_cfg = dict(average_clips='prob', max_testing_views = 10)  
)

# dataset settings
dataset_type = "VideoDataset"
data_root = "/home/varun/datasets/soccer_data_full/"
data_root_val = "/home/varun/datasets/soccer_data_full/"
ann_file_train = "/home/varun/datasets/annot_train.txt"
ann_file_val = "/home/varun/datasets/annot_val.txt"
ann_file_test = "/home/varun/datasets/annot_test.txt"

#normalization values for preprocessing in pipeline
img_norm_cfg = dict(
    mean=[110.2008, 100.63983, 95.99475],
    std=[58.14765, 56.46975, 55.332195],
    to_bgr=False,
)

#pipelines for training, val and test
train_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=16, frame_interval=2, num_clips=1), #32 frames, every 2 frames, 1 clip
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames", clip_len=32, frame_interval=2, num_clips=1, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
test_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames", clip_len=32, frame_interval=2, out_of_bound_opt="repeat_first_last", test_mode = True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="ThreeCrop", crop_size=256),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
    ),
)

gpu_ids=[0]
#pretrained model loading
load_from = "checkpoints/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth"
work_dir = "./work_dirs/ipcsn_sports1m_pretrained_soccerdatafull"  # dir to store results, logs

# Learning policy

optimizer = dict(type='Adam', lr=0.0000002,betas=(0.9, 0.999))

# Learning rate config
lr_config = dict(
        policy = 'CosineAnnealing',
        min_lr_ratio = 5e-5,
        warmup = 'linear',
        warmup_ratio = 0.1,
        warmup_iters = 1,
        warmup_by_epoch=True,
        )
# Evaluation config, interval = interval of validation done
evaluation = dict(
        interval = 1,
        metric_options = dict(top_k_accuracy = dict(topk = (1,2))),
        )
eval_config = dict(
        metric_options = dict(top_k_accuracy = dict(topk = (1,2))),
        )

total_epochs = 6

#How often checkpoint file is stored
checkpoint_config = dict(interval = 1)



