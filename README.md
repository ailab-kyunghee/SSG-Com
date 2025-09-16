# **Towards Holistic Surgical Scene Graph**
This repository contains the official implementation of our MICCAI 2025 paper:
Towards Holistic Surgical Scene Graph 

## Installation
Follow the installation instructions in [![LG-CVS](https://github.com/CAMMA-public/SurgLatentGraph/tree/main)]

📂 Dataset: Endoscapes-SG201
We introduce Endoscapes-SG201, an extension of the Endoscapes dataset.
Endoscapes-SG201 provides:
•	✅ Refined instrument annotations (6 instrument sub-classes: Hook, Grasper, Clipper, Bipolar, Irrigator, Scissors).
•	✅ Triplet (Instrument–Verb–Target Anatomy) annotations.
•	✅ Hand identity labels (Operator's right, left, assistant).

Download the Endoscapes dataset from [![Endoscapes Dataset](https://img.shields.io/badge/Endoscapes-Dataset%20+%20Annotations-red)](https://github.com/CAMMA-public/Endoscapes)

#Download 
[![Endoscapes Dataset](https://img.shields.io/badge/Endoscapes-Dataset%20+%20Annotations-red)]()

The final directory structure should be as follows:
```shell
data/mmdet_datasets
└── endoscapes/
    └── train/
        └── 1_14050.jpg
        ...
        └── 120_40750.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
        └── train_endo_with_tri_annotations_coco.json
    └── val/
        └── 121_23575.jpg
        ...
        └── 161_39400.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
        └── val_endo_with_tri_annotations_coco.json
    └── test/
        └── 162_1225.jpg
        ...
        └── 201_55250.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
        └── test_endo_with_tri_annotations_coco.json
```