import os
import copy

_base_ = os.path.expandvars('$MMDETECTION/configs/_base_/datasets/coco_instance.py')
custom_imports = dict(imports=['datasets.custom_loading'], allow_failed_imports=False)

# Modify dataset related settings

# data_root='data/mmdet_datasets/endoscapes'
data_root='/local_datasets/endoscapes'
metainfo = {
    'classes': ('cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct',
        'gallbladder', 'clipper','bipolar','grasper', 'scissors','hook', 'irrigator' ),
    'palette': [(255, 255, 100), (102, 178, 255), (255, 0, 0), (0, 102, 51), (51, 255, 103), (255, 151, 53), 
                (123, 45, 67),(200, 150, 80), (80, 200, 180),  (40, 160, 250),(10, 100, 90) ]
}

rand_aug_surg = [
        [dict(type='ShearX', level=8)],
        [dict(type='ShearY', level=8)],
        [dict(type='Rotate', level=8)],
        [dict(type='TranslateX', level=8)],
        [dict(type='TranslateY', level=8)],
        [dict(type='AutoContrast', level=8)],
        [dict(type='Equalize', level=8)],
        [dict(type='Contrast', level=8)],
        [dict(type='Color', level=8)],
        [dict(type='Brightness', level=8)],
        [dict(type='Sharpness', level=8)],
]

train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotationsWithDS_tri',
            with_bbox=True,
            # with_mask=True,
            with_mask=False,
       
        ),
        dict(
            type='Resize',
            scale=(399, 224),
            keep_ratio=True,
        ),
        dict(
            type='RandomFlip',
            prob=0.5,
        ),
        dict(
            type='RandAugment',
            aug_space=rand_aug_surg,
        ),
        dict(
            type='Color',
            min_mag = 0.6,
            max_mag = 1.4,
        ),
        #dict(
        #    type='RandomErasing',
        #    n_patches=(0, 1),
        #    ratio=(0.3, 1),
        #),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
        dict(type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'flip', 'flip_direction', 'homography_matrix', 'ds', 'is_det_keyframe', 'video_id', 'oper','triplet')
        ),
]

eval_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='Resize',
            scale=(399, 224),
            keep_ratio=True,
        ),
        dict(type='LoadAnnotationsWithDS_tri',
            with_bbox=True,
            # with_mask=True,
            with_mask=False,
        
            ),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'ds', 'is_det_keyframe', 'video_id', 'oper','triplet'),
        ),
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS_tri',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_ds_coco_triplet.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
    ),
    batch_sampler=dict(drop_last=True),
)

train_eval_dataloader = copy.deepcopy(_base_.val_dataloader)
train_eval_dataloader['dataset'].update(dict(
        type='CocoDatasetWithDS_tri',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_ds_coco_triplet.json',
        data_prefix=dict(img='train/'),
        pipeline=eval_pipeline,
    )
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS_tri',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_ds_coco_triplet.json',
        data_prefix=dict(img='val/'),
        pipeline=eval_pipeline))

test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        type='CocoDatasetWithDS_tri',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/annotation_ds_coco_triplet.json',
        data_prefix=dict(img='test/'),
        pipeline=eval_pipeline))

# metric
val_evaluator = dict(ann_file=os.path.join(data_root, 'val/annotation_ds_coco_triplet.json'), format_only=False,)
test_evaluator = dict(ann_file=os.path.join(data_root, 'test/annotation_ds_coco_triplet.json'), format_only=False,)
evaluation = dict(metric=['bbox', 'segm'])
