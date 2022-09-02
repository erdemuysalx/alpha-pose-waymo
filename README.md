
# Utility scripts for using AlphaPose on Waymo OD

* waymo_detector.py: WaymoDetectionLoader class which loads TFRecord files and store necessary information i.e. label, bbox etc. and crops the images.

* waymo_inference.py: Runs inference on cropped images comes from WaymoDetectionLoader

## Requirements

 - https://github.com/MVIG-SJTU/AlphaPose/
 - https://github.com/waymo-research/waymo-open-dataset

## Installation

    pip install requirements.txt

## Running

    python waymo_inference.py --cfg /path_to/256x192_res50_lr1e-3_1x.yaml --checkpoint /path_to/fast_res50_256x192.pth --waymo /path_to/waymo_open_dataset

*PS: All the arguments belongs to AlphaPose except the "--waymo" which indicates the path to the waymo dataset*

### Default paths to required arguments

--cfg AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 
--checkpoint AlphaPose/pretrained_models/fast_res50_256x192.pth

Details https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#models

## Problems

 - [ ] Verification of estimated key points by visualization
 - [ ] Custom format for writing key points
