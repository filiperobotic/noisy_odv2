# dataset settings
dataset_type = 'VOCDataset'
# dataset_type = 'VOCDatasetPartial'

# -----------------------------------------------------------------------------
# Filipe: Switch VOC annotation variants (clean / noisy) WITHOUT copying XMLs.
#
# You keep ONE canonical VOCdevkit with images + ImageSets:
#   <VOC_BASE_ROOT>/VOC2007/JPEGImages, ImageSets
#   <VOC_BASE_ROOT>/VOC2012/JPEGImages, ImageSets
#
# And you keep noisy XMLs elsewhere (your current layout), e.g.:
#   .../NoiseAnnotations/simetrics/
#       Annotations_VOC2007_trainval_class_noise_perc_20_simetric/*.xml
#       Annotations_VOC2012_trainval_class_noise_perc_20_simetric/*.xml
#
# At runtime we create a lightweight "virtual VOCdevkit" with symlinks:
#   JPEGImages  -> canonical JPEGImages
#   ImageSets   -> canonical ImageSets
#   Annotations -> chosen noisy-XML folder
#
# How to control via env vars:
#   # 1) Point to canonical VOCdevkit (with images)
#   export VOC_BASE_ROOT=/home/pesquisador/pesquisa/filipe/noisy_odv2/data/VOCdevkit
#
#   # 2) Choose annotation source:
#   export VOC_ANN_VARIANT=clean
#   # OR (recommended with your current folders):
#   export VOC_ANN_VARIANT=custom
#   export VOC2007_ANN_DIR=/home/.../NoiseAnnotations/simetrics/Annotations_VOC2007_trainval_class_noise_perc_20_simetric
#   export VOC2012_ANN_DIR=/home/.../NoiseAnnotations/simetrics/Annotations_VOC2012_trainval_class_noise_perc_20_simetric
#
# Optional:
#   export VOC_VIRTUAL_ROOT=/home/.../noisy_odv2/data/VOCdevkit_variants
# -----------------------------------------------------------------------------
import os
from pathlib import Path

VOC_BASE_ROOT = Path(os.getenv('VOC_BASE_ROOT', 'data/VOCdevkit/')).expanduser().resolve()
VOC_ANN_VARIANT = os.getenv('VOC_ANN_VARIANT', 'clean')  # clean | custom
VOC_VIRTUAL_ROOT = Path(os.getenv('VOC_VIRTUAL_ROOT', 'data/VOCdevkit_variants/')).expanduser().resolve() / VOC_ANN_VARIANT


def _symlink_dir(src: Path, dst: Path):
    """Create/refresh a directory symlink dst -> src."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.is_symlink():
            dst.unlink()
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    except FileExistsError:
        pass


def _prepare_virtual_vocdevkit() -> Path:
    """Return a VOCdevkit root that MMDetection should use."""
    if VOC_ANN_VARIANT == 'clean':
        return VOC_BASE_ROOT

    if VOC_ANN_VARIANT != 'custom':
        raise ValueError(
            f"Unsupported VOC_ANN_VARIANT={VOC_ANN_VARIANT!r}. Use 'clean' or 'custom'."
        )

    ann_2007 = os.getenv('VOC2007_ANN_DIR', '').strip()
    ann_2012 = os.getenv('VOC2012_ANN_DIR', '').strip()
    if not ann_2007 or not ann_2012:
        raise RuntimeError(
            "VOC_ANN_VARIANT=custom requires BOTH env vars: VOC2007_ANN_DIR and VOC2012_ANN_DIR"
        )

    ann_2007 = Path(ann_2007).expanduser().resolve()
    ann_2012 = Path(ann_2012).expanduser().resolve()

    # Sanity checks
    for year, ann_dir in [('VOC2007', ann_2007), ('VOC2012', ann_2012)]:
        base_year = VOC_BASE_ROOT / year
        if not (base_year / 'JPEGImages').exists():
            raise FileNotFoundError(f"Missing canonical JPEGImages: {base_year / 'JPEGImages'}")
        if not (base_year / 'ImageSets').exists():
            raise FileNotFoundError(f"Missing canonical ImageSets: {base_year / 'ImageSets'}")
        if not ann_dir.exists():
            raise FileNotFoundError(f"Missing noisy annotation folder for {year}: {ann_dir}")

    # Create virtual VOC2007/VOC2012
    for year, ann_dir in [('VOC2007', ann_2007), ('VOC2012', ann_2012)]:
        base_year = VOC_BASE_ROOT / year
        virt_year = VOC_VIRTUAL_ROOT / year
        virt_year.mkdir(parents=True, exist_ok=True)

        _symlink_dir(base_year / 'JPEGImages', virt_year / 'JPEGImages')
        _symlink_dir(base_year / 'ImageSets', virt_year / 'ImageSets')
        _symlink_dir(ann_dir, virt_year / 'Annotations')

    return VOC_VIRTUAL_ROOT


# This is what MMDetection will use as the dataset root
data_root = str(_prepare_virtual_vocdevkit()) + '/'


# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically Infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/segmentation/VOCdevkit/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#         'data/': 's3://openmmlab/datasets/segmentation/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5), #debug
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(

    # batch_size=2,
    batch_size=16,
    num_workers=4,
    # num_workers=0,
    # persistent_workers=True,
    persistent_workers=False,   # FILIPE DEBUG
    sampler=dict(type='DefaultSampler', shuffle=True),
    # sampler=dict(type='DefaultSampler', shuffle=True, seed=2025), #FILIPE DEBUG
    # sampler=dict(type='DefaultSampler', shuffle=False), #FILIPE DEBUG
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    pin_memory=True,   
    #pin_memory=False,    # FILIPE DEBUGGING
    dataset=dict(
        type='RepeatDataset',
        #times=3,
        times=1,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    #ann_file='VOC2007/ImageSets/Main/trainval_debug_nano.txt',  # head -n 10 trainval.txt_ > trainval_debug_nano.txt 
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    serialize_data=False,  # Define como False (Filipe)
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                    pipeline=train_pipeline,
                    backend_args=backend_args),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    serialize_data=False,  # Define como  [FILIPE]
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
                    pipeline=train_pipeline,
                    backend_args=backend_args)
            ])))


# debug_train_dataloader = dict(

#     batch_size=2,
#     # batch_size=8,
#     num_workers=2,
#     persistent_workers=True,
#     # persistent_workers=False,   # FILIPE DEBUG
#     #sampler=dict(type='DefaultSampler', shuffle=True),
#     sampler=dict(type='DefaultSampler', shuffle=False), #FILIPE DEBUG
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     pin_memory=True,   
#     #pin_memory=False,    # FILIPE DEBUGGING
#     dataset=dict(
#         type='RepeatDataset',
#         #times=3,
#         times=1,
#         dataset=dict(
#             type='ConcatDataset',
#             # VOCDataset will add different `dataset_type` in dataset.metainfo,
#             # which will get error if using ConcatDataset. Adding
#             # `ignore_keys` can avoid this error.
#             ignore_keys=['dataset_type'],
#             datasets=[
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2007/ImageSets/Main/trainval.txt',
#                     #ann_file='VOC2007/ImageSets/Main/trainval_debug_nano.txt',  # head -n 10 trainval.txt_ > trainval_debug_nano.txt 
#                     data_prefix=dict(sub_data_root='VOC2007/'),
#                     # test_mode=True,
#                     # serialize_data=False,  # Define como False (Filipe)
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
#                     pipeline=train_pipeline,
#                     backend_args=backend_args),
#                 dict(
#                     type=dataset_type,
#                     data_root=data_root,
#                     ann_file='VOC2012/ImageSets/Main/trainval.txt',
#                     data_prefix=dict(sub_data_root='VOC2012/'),
#                     # test_mode=True,
#                     # serialize_data=False,  # Define como  [FILIPE]
#                     filter_cfg=dict(
#                         filter_empty_gt=True, min_size=0, bbox_min_size=0), # MIN_SIZE & BBOX_MIN_SIZE ALTERADOS
#                     pipeline=train_pipeline,
#                     backend_args=backend_args)
#             ])))

# debug_train_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='VOC2007/ImageSets/Main/trainval.txt',
#         data_prefix=dict(sub_data_root='VOC2007/'),
#         test_mode=True,
#         #pipeline=test_pipeline,
#         pipeline=train_pipeline,
#         backend_args=backend_args))

val_dataloader = dict(
    batch_size=6,
    num_workers=2,
    # num_workers=0,
    #persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader
# test_dataloader = debug_train_dataloader #debug FILIPE
# val_dataloader = debug_train_dataloader  # debug FILIPE

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator 
