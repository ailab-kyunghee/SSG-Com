import os
import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base_triplet_full_pplus.py',
    os.path.expandvars('/data/cekkec/project/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]

# extract detector, data preprocessor config from base
detector = copy.deepcopy(_base_.model)
detector.roi_head.bbox_head.num_classes = _base_.num_classes
detector.test_cfg.rcnn.max_per_img = _base_.num_nodes
dp = copy.deepcopy(_base_.model.data_preprocessor)
dp.pad_size_divisor = 1
del _base_.model
del detector.data_preprocessor

# extract lg config, set detector
model = copy.deepcopy(_base_.lg_model)
model.data_preprocessor = dp
model.detector = detector
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.sem_feat_use_masks = False

# trainable bb, neck
model.trainable_backbone_cfg=copy.deepcopy(detector.backbone)
model.trainable_backbone_cfg.frozen_stages=_base_.trainable_backbone_frozen_stages
if 'neck' in detector:
    model.trainable_neck_cfg=copy.deepcopy(detector.neck)

del _base_.lg_model

# modify load_from
load_from = _base_.load_from.replace('base', 'faster_rcnn_pplus')
train_dataloader =dict(dataset=dict(ann_file='train/ds_triplet_train_pplus_annotations_coco_final_full.json'))
train_eval_dataloader =dict(dataset=dict(ann_file='train/ds_triplet_train_pplus_annotations_coco_final_full.json'))
val_dataloader =dict(dataset=dict(ann_file='val/ds_triplet_val_pplus_annotations_coco_final_full.json'))
test_dataloader =dict(dataset=dict(ann_file='test/ds_triplet_test_pplus_annotations_coco_final_full.json'))
train_evaluator = dict(ann_file='/local_datasets/endoscapes/train/ds_triplet_train_pplus_annotations_coco_final_full.json', num_classes=34,)
val_evaluator = dict(ann_file='/local_datasets/endoscapes/val/ds_triplet_val_pplus_annotations_coco_final_full.json', num_classes=34,)
test_evaluator = dict(ann_file='/local_datasets/endoscapes/test/ds_triplet_test_pplus_annotations_coco_final_full.json', num_classes=34,)