custom_imports = dict(imports=["elastic_vit_ss.models"], allow_failed_imports=False)

dataset_type = "ADE20KDataset"
data_root = "data/ADEChallengeData2016"
crop_size = (512, 512)
backend_args = None
norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", reduce_zero_label=True, backend_args=backend_args),
    dict(type="RandomResize", scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(2048, 512), keep_ratio=True),
    dict(type="LoadAnnotations", reduce_zero_label=True, backend_args=backend_args),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, data_prefix=dict(img_path="images/training", seg_map_path="annotations/training"), pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, data_prefix=dict(img_path="images/validation", seg_map_path="annotations/validation"), pipeline=test_pipeline),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(type="SegDataPreProcessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True, pad_val=0, seg_pad_val=255, size=crop_size),
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
    decode_head=dict(
        type="UPerHead",
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=160000, val_interval=16000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False),
]
optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01), clip_grad=dict(max_norm=1.0, norm_type=2))
default_scope = "mmseg"
default_hooks = dict(timer=dict(type="IterTimerHook"), logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False), param_scheduler=dict(type="ParamSchedulerHook"), checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=16000), sampler_seed=dict(type="DistSamplerSeedHook"), visualization=dict(type="SegVisualizationHook"))
env_cfg = dict(cudnn_benchmark=True, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

