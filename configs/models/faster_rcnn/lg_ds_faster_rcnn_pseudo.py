import os
import copy

# modify base for different detectors
_base_ = [
    '../lg_ds_base_pseudo_.py',
    os.path.expandvars('/data/cekkec/project/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]
# configs/models/lg_ds_base_pseudo_.py
# extract detector, data preprocessor config from base


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
load_from = _base_.load_from.replace('base', 'faster_rcnn')


train_dataloader =dict(dataset=dict(ann_file='train/annotation_ds_coco.json'))
train_eval_dataloader =dict(dataset=dict(ann_file='train/annotation_ds_coco.json'))
val_dataloader =dict(dataset=dict(ann_file='val/annotation_ds_coco.json'))
test_dataloader =dict(dataset=dict(ann_file='test/annotation_ds_coco.json'))
train_evaluator = dict(ann_file='/local_datasets/endoscapes/val/annotation_ds_coco.json')
val_evaluator = dict(ann_file='/local_datasets/endoscapes/val/annotation_ds_coco.json')
test_evaluator = dict(ann_file='/local_datasets/endoscapes/test/annotation_ds_coco.json')
