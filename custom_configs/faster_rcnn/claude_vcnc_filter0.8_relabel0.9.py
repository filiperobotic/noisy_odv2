"""
Exemplo de configuração para usar o Visual Clustering Noise Correction Hook.

Para usar, adicione este hook na configuração do seu experimento MMDetection.
"""

_base_ = [
    # '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/models/faster-rcnn_r50_fpn_20c.py',
    #   '../_base_/datasets/voc0712_corrigido_debug.py',
    '../_base_/datasets/voc0712_corrigido_v2_rep1.py',
    '../_base_/default_runtime.py'
]

env_cfg = dict(cudnn_benchmark=False)

custom_imports = dict(
    imports=['custom_configs.hooks.wandb_pred_buckets_hook'],
    allow_failed_imports=False
    

)

custom_imports = dict(
    imports=['custom_configs.hooks.vcnc_complete_hook',
             'custom_configs.hooks.wandb_pred_buckets_hook'],  # Caminho correto para o arquivo do hook
    allow_failed_imports=False
)

custom_hooks = [
    dict(
    type='VCNCCompleteHook',
    
    # Configuração geral
    warmup_epochs=1,
    num_classes=20,
    
    # === ETAPA 1: Relabel por confiança (Baseline) ===
    enable_confidence_relabel=True,
    relabel_confidence_threshold=0.9,  # Igual ao baseline
    
    # === ETAPA 2: Clustering Visual (VCNC) ===
    enable_clustering_relabel=True,
    n_clusters=150,
    use_softmax_as_embedding=True,
    
    # Critérios progressivos
    progressive_epochs=4,
    
    # Conservador (épocas 2-4)
    early_anchor_gmm_threshold=0.15,
    early_anchor_pred_agreement=0.85,
    early_anchor_confidence=0.9,
    early_suspect_gmm_threshold=0.8,
    early_similarity_threshold=0.7,
    early_cluster_consensus=0.85,
    
    # Agressivo (épocas 5+)
    anchor_gmm_threshold=0.4,
    anchor_pred_agreement=0.6,
    anchor_confidence=0.7,
    suspect_gmm_threshold=0.5,
    similarity_threshold=0.4,
    cluster_consensus=0.6,
    
    # === ETAPA 3: Filtragem GMM (Baseline) ===
    enable_gmm_filter=True,
    gmm_components=4,
    filter_gmm_threshold=0.8,  # p_noise > 0.7 → ignore
    
    # Outros
    iou_assigner=0.5,
    reload_dataset=True,
    debug=True
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






# ============================================================
# VARIAÇÕES PARA TESTAR
# ============================================================

# Versão conservadora (menos relabels, mais segura)
# vcnc_conservative = dict(
#     type='VisualClusteringNoiseCorrectionHook',
#     warmup_epochs=2,
#     num_classes=20,
#     n_clusters=100,
#     anchor_gmm_threshold=0.2,     # Mais restritivo
#     anchor_pred_agreement=0.8,    # Mais restritivo
#     anchor_confidence=0.85,       # Mais restritivo
#     suspect_gmm_threshold=0.7,    # Mais restritivo
#     cluster_consensus=0.8,        # Mais consenso necessário
#     similarity_threshold=0.6,     # Mais similaridade necessária
#     gmm_components=4,
#     enable_spatial_refinement=True,
#     reload_dataset=True,
#     debug=True
# )

# Versão agressiva (mais relabels)
# vcnc_aggressive = dict(
#     type='VisualClusteringNoiseCorrectionHook',
#     warmup_epochs=1,
#     num_classes=20,
#     n_clusters=150,               # Mais clusters
#     anchor_gmm_threshold=0.4,     # Menos restritivo
#     anchor_pred_agreement=0.6,    # Menos restritivo
#     anchor_confidence=0.7,        # Menos restritivo
#     suspect_gmm_threshold=0.5,    # Menos restritivo
#     cluster_consensus=0.6,        # Menos consenso
#     similarity_threshold=0.4,     # Menos similaridade
#     gmm_components=4,
#     enable_spatial_refinement=True,
#     reload_dataset=True,
#     debug=True
# )

# ============================================================
# MÉTRICAS PARA MONITORAR
# ============================================================

"""
Durante o treinamento, observe:

1. Taxa de relabel por época:
   - Muito baixa (<1%): critérios muito restritivos
   - Muito alta (>10%): critérios muito permissivos
   - Ideal: 2-5%

2. mAP por época:
   - Deve subir consistentemente
   - Comparar com baseline GMM+Relabel

3. Classes mais afetadas:
   - Verificar se classes problemáticas (boat, pottedplant, chair) melhoram

4. Distribuição por cluster:
   - Clusters muito pequenos: aumentar n_clusters
   - Clusters muito grandes: reduzir n_clusters
"""
