# _base_ = [
#     '../_base_/models/faster-rcnn_r50_fpn.py', '../_base_/datasets/voc0712_corrigido.py',
#     '../_base_/default_runtime.py'
# ]
_base_ = [
    # '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/models/faster-rcnn_r50_fpn_20c.py',
    #   '../_base_/datasets/voc0712_corrigido_debug.py',
    '../_base_/datasets/voc0712_corrigido_v2_rep1.py',
    '../_base_/default_runtime.py'
]

        

# my parameters
# warmup_epochs = 1
thr_noise = 0.9 #nao esta sendo usado na pratica


# thresholds

use_percentile = False
percentile = 80.0 #nao esta sendo usado na pratica

do_relabel = False

num_classes = 20
reload_dataset = True
iou_assigner = 0.5
low_quality = True
# train_ratio=0.5
relabel_thr_ctx = 0.8
relabel_thr_high = 0.7




custom_imports = dict(
    imports=[
             'custom_configs.hooks.graph_relabel_hook_v2',
             'custom_configs.hooks.wandb_pred_buckets_hook'],  # hook v2 com ConG/KLD
    allow_failed_imports=False
)




custom_hooks = [
   
    dict(type='MyHookGraphNoiseRelabel', priority='NORMAL',
        #  warmup_epochs = warmup_epochs,
         thr_noise = thr_noise,
         use_percentile = use_percentile,
         percentile = percentile,
         do_relabel = do_relabel,
         num_classes = num_classes,
         reload_dataset = reload_dataset,
         iou_assigner = iou_assigner,
         low_quality = low_quality,
        #  train_ratio = train_ratio,
         relabel_thr_ctx = relabel_thr_ctx,
         relabel_thr_high = relabel_thr_high,
         # parâmetros do ConG (GCRN-like)
         cong_hidden = 128,
         cong_lr = 1e-3,
         cong_train_steps = 100,
         cong_alpha = 0.5,
         use_wandb=True,
        wandb_project='NOD',
        # wandb_run_name='gcnc-filtering-0.9',
        wandb_max_images=16,
    ),

         
 
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


model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

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

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
