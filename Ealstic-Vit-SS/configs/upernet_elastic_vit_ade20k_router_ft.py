_base_ = "./upernet_elastic_vit_ade20k.py"

model = dict(
    backbone=dict(
        freeze_vit=True,
        freeze_router=False,
    )
)

# Load a dense-task checkpoint with trained UPerNet heads via --cfg-options load_from=...
# Only backbone.router keeps a nonzero learning rate in this fine-tuning phase.
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.patch_embed": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.blocks": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.norm": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.cls_token": dict(lr_mult=0.0, decay_mult=0.0),
            "backbone.pos_embed": dict(lr_mult=0.0, decay_mult=0.0),
            "decode_head": dict(lr_mult=0.0, decay_mult=0.0),
            "auxiliary_head": dict(lr_mult=0.0, decay_mult=0.0),
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
train_cfg = dict(type="IterBasedTrainLoop", max_iters=16000, val_interval=4000)

