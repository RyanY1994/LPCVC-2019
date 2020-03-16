# Dataset Preparation
1. Download [COCO-2017 dataset](http://cocodataset.org/#download) for Object Detection Task, in ./data/coco-2017.

2. Run genMinval8000.py in ./data/coco/coco-ovic to generate annotation files (i.e., ./annotations/instances_train_L.json and ./annotations/instances_minval_L.json) for training and validataion according to ovic_val_2017_list.txt required by [OVIC](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt).

3. Run splitImage.py in ./data/coco/coco-ovic to split images according to ovic_val_2017_list.txt. The splitted images are stored in ./data/coco/coco-ovic/val_ovic and ./data/coco/coco-ovic/train_ovic.

4. Run removeSmallObj.py in ./data/coco/coco-ovic to generate annotation file (i.e., ./annotations/instances_train_L_256.json) for images with area larger or equal to 256 for the removal of small objects.
