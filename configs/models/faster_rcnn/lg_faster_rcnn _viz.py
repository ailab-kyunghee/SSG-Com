import copy
import os

_base_=['../lg_base_box.py',
    os.path.expandvars('/data/cekkec/project/mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py'),
]
# custom imports to register custom modules - extend base imports
orig_imports = _base_.custom_imports.imports
custom_imports = dict(imports=orig_imports + ['visualizer.LatentGraphVisualizer'], allow_failed_imports=False)

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
del _base_.lg_model

# modify load_from
load_from = 'weights/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c_LG.pth'

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
)
auto_scale_lr = dict(enable=True)
# visualization configuration
visualizer = dict(
    type='LatentGraphVisualizer',
    dataset='endoscapes',
    detector='faster_rcnn',
    draw=True,                    # Enable visualization
    save_graphs=True,             # Save graph data as .npz files
    data_prefix='lg_cvs_score_thr_0.4',
    results_dir='results',
    gt_graph_use_pred_instances=True,  # Use predicted instances for GT graph
    vis_backends=[dict(type='LocalVisBackend')]
)
visualization = dict(
            draw=True,
        score_thr=0.4,
        show=False,
        type='DetVisualizationHook'
    )
# visualization hook configuration
# default hooks to ensure visualization works during testing
default_hooks = dict(
    visualization=visualization
)


train_dataloader =dict(dataset=dict(ann_file='train/annotation_coco.json'))
train_eval_dataloader =dict(dataset=dict(ann_file='train/annotation_coco.json'))
val_dataloader =dict(dataset=dict(ann_file='val/annotation_coco.json'))
test_dataloader =dict(dataset=dict(ann_file='test/annotation_coco.json'))
val_evaluator = dict(ann_file='/local_datasets/endoscapes/val/annotation_coco.json')
test_evaluator = dict(ann_file='/local_datasets/endoscapes/test/annotation_coco.json')