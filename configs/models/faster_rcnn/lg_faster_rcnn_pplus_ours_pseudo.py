import copy
import os

_base_=['../lg_base_box_pplus.py',
    os.path.expandvars('/data/cekkec/project/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]
#configs/models/lg_base_box_pplus.py
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
model.reconstruction_img_stats=dict(mean=dp.mean, std=dp.std)
model.detector = detector
model.roi_extractor = copy.deepcopy(detector.roi_head.bbox_roi_extractor)
model.roi_extractor.roi_layer.output_size = 1
model.graph_head.type='GraphHead_tri'
del _base_.lg_model

# modify load_from
load_from = 'weights/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c_LG.pth'

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
)
auto_scale_lr = dict(enable=True)
model.graph_head.triplet_loss_weight=0.2
model.graph_head.num_triplet_edge_classes=5 # data_label_new/train_endo_sg201_annotations_coco_pseudo_top_350.json
# train_dataloader =dict(dataset=dict(ann_file='train/train_endo_with_tri_annotations_coco.json'))
# train_eval_dataloader =dict(dataset=dict(ann_file='train/train_endo_with_tri_annotations_coco.json'))
# val_dataloader =dict(dataset=dict(ann_file='val/val_endo_with_tri_annotations_coco.json'))
# test_dataloader =dict(dataset=dict(ann_file='test/test_endo_with_tri_annotations_coco.json'))
# val_evaluator = dict(ann_file='/local_datasets/endoscapes/val/val_endo_with_tri_annotations_coco.json')
# test_evaluator = dict(ann_file='/local_datasets/endoscapes/test/test_endo_with_tri_annotations_coco.json')
train_dataloader =dict(dataset=dict(ann_file='train/train_endo_sg201_annotations_coco_pseudo_top_350.json'))
train_eval_dataloader =dict(dataset=dict(ann_file='train/train_endo_sg201_annotations_coco_pseudo_top_350.json'))
val_dataloader =dict(dataset=dict(ann_file='val/val_endo_sg201_annotations_coco_pseudo_top_350.json'))
test_dataloader =dict(dataset=dict(ann_file='test/test_endo_sg201_annotations_coco_pseudo_top_350.json'))
val_evaluator = dict(ann_file='/local_datasets/endoscapes/val/val_endo_sg201_annotations_coco_pseudo_top_350.json')
test_evaluator = dict(ann_file='/local_datasets/endoscapes/test/test_endo_sg201_annotations_coco_pseudo_top_350.json')