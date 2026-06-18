_base_ = "./cascade_mask_rcnn_elastic_vit_coco.py"

model = dict(
    backbone=dict(
        freeze_vit=True,
        freeze_router=False,
    )
)

# Load a dense-task checkpoint with trained detector heads via --cfg-options load_from=...
# Only backbone.router keeps a nonzero learning rate in this fine-tuning phase.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.patch_embed": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.blocks": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.norm": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.cls_token": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.pos_embed": dict(lr_mult=0.0, decay_mult=0.0),
            "neck": dict(lr_mult=0.0, decay_mult=0.0),
            "rpn_head": dict(lr_mult=0.0, decay_mult=0.0),
            "roi_head": dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=3, val_interval=1)

