custom_imports = dict(imports=["elastic_vit_od.models"], allow_failed_imports=False)

dataset_type = "CocoDataset"
data_root = "data/coco/"
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]
test_dev_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(type=dataset_type, data_root=data_root, ann_file="annotations/instances_train2017.json", data_prefix=dict(img="train2017/"), filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline, backend_args=backend_args),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, ann_file="annotations/instances_val2017.json", data_prefix=dict(img="val2017/"), test_mode=True, pipeline=test_pipeline, backend_args=backend_args),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, ann_file="annotations/image_info_test-dev2017.json", data_prefix=dict(img="test2017/"), test_mode=True, pipeline=test_dev_pipeline, backend_args=backend_args),
)

val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/instances_val2017.json", metric="bbox")
test_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/image_info_test-dev2017.json", metric="bbox", format_only=True, outfile_prefix="./work_dirs/reppoints_v2_elastic_vit_coco/test-dev")

model = dict(
    type="RepPointsDetector",
    data_preprocessor=dict(type="DetDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_size_divisor=32),
    backbone=dict(
        type="ElasticViTBackbone",
        model_name="vit_base_patch16_224",
        pretrained=False,
        checkpoint_path=None,
        router_checkpoint=None,
        router_input_mode="both",
        router_hidden_dim=64,
        mlp_choices=[768, 1536, 2304, 3072],
        head_choices=[3, 6, 9, 12],
        out_indices=(2, 5, 8, 11),
        pyramid_scales=(4.0, 2.0, 1.0, 0.5),
    ),
    neck=dict(type="FPN", in_channels=[768, 768, 768, 768], out_channels=256, start_level=1, add_extra_convs="on_input", num_outs=5),
    bbox_head=dict(
        type="RepPointsHead",
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox_init=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=0.5),
        loss_bbox_refine=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
        transform_method="moment",
    ),
    train_cfg=dict(
        init=dict(assigner=dict(type="PointAssigner", scale=4, pos_num=1), allowed_border=-1, pos_weight=-1, debug=False),
        refine=dict(assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0, ignore_iof_thr=-1), allowed_border=-1, pos_weight=-1, debug=False),
    ),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.05), clip_grad=dict(max_norm=1.0, norm_type=2))
default_scope = "mmdet"
default_hooks = dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50), param_scheduler=dict(type="ParamSchedulerHook"), checkpoint=dict(type="CheckpointHook", interval=1), sampler_seed=dict(type="DistSamplerSeedHook"), visualization=dict(type="DetVisualizationHook"))
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False

