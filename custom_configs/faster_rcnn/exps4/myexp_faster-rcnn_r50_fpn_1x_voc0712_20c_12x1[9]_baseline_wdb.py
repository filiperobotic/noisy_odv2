
_base_ = [
    # '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/models/faster-rcnn_r50_fpn_20c.py',
    '../_base_/datasets/voc0712_corrigido_v2_rep1.py',
    '../_base_/default_runtime.py'
]

env_cfg = dict(cudnn_benchmark=False)

# Permitir importar o hook
custom_imports = dict(
    imports=['custom_configs.hooks.wandb_pred_buckets_hook'],
    allow_failed_imports=False
    
)

# Registra o hook
custom_hooks = [
    dict(
        type='WandbPredBucketsHook',
        high_conf_thr=0.9,   # bucket de alto-confiável
        min_pred_thr=0.5,    # limiar dos demais buckets
        iou_thr=0.5,
        num_images=16
    )
]


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(project='NOD')),
]



visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')



model = dict(roi_head=dict(
    # type='StandardRoIHeadWithClsScores',   # <-- usar a head nova
    bbox_head=dict(num_classes=20)))

# training schedule, voc dataset is repeated 3 times, in
# `_base_/datasets/voc0712.py`, so the actual epoch = 4 * 3 = 12
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
max_epochs
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
