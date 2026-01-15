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
reload_dataset = True
relabel_conf = 0.9
filter_conf = 0.95
filter_warmup= 2
iou_assigner = 0.5
low_quality = True
filter_thr = 0.7
numGMM = 4
filter_type = 'pred'
# overlap_filter_epochs=5
overlap_cont_thr=0.7
overlap_filter_start=6,    # Começa na época 6
overlap_filter_end=10,     # Termina na época 10

#












custom_imports = dict(
    imports=['custom_configs.hooks.sample_curr_relabeling_hook_v2',
             'custom_configs.hooks.wandb_pred_buckets_hook'],  # Caminho correto para o arquivo do hook
    allow_failed_imports=False
)




custom_hooks = [
   
    dict(type='MyHookReverseCurrIntoFilterPredGT_Class_Relabel', priority='NORMAL',
         reload_dataset = reload_dataset,
         relabel_conf = relabel_conf,
         filter_conf = filter_conf,
         filter_warmup = filter_warmup,
         iou_assigner = iou_assigner,
         low_quality = low_quality,
         filter_thr = filter_thr,
         numGMM = numGMM,
        #   overlap_filter_epochs=overlap_filter_epochs,
         overlap_filter_start=overlap_filter_start,    # Começa na época 6
         overlap_filter_end=overlap_filter_end,     # Termina na época 10
          overlap_iou_thr=overlap_cont_thr
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
