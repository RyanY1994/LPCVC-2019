import os 
import numpy as np
import json
import sys 

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask


out_root = "/data2/chengxiang/models/research/object_detection/data/coco/coco-ovic"
minval_json = os.path.join(out_root, "annotations/instances_minval_L.json")
train_json = os.path.join(out_root, "annotations/instances_train_L.json")


# Validation Dataset
minval_f = open(minval_json,"r")
minval_dict = json.load(minval_f)
minval_f.close()
_COCO = COCO(minval_json)

num = 0
num_128 = 0
num_256 = 0
num_512 = 0
num_1024 = 0
for i, item in enumerate(minval_dict["images"]): # 7991 images
	if(i % 1000==0):
		print(i)
	index_img = item["id"]
	annIds = _COCO.getAnnIds(imgIds = index_img, iscrowd = None)
	objs = _COCO.loadAnns(annIds)
	for obj in objs:
		num += 1
		if(float(obj['area'])>=128):
			num_128 += 1
		if(float(obj['area'])>=256):
			num_256 += 1
		if(float(obj['area'])>=512):
			num_512 += 1
		if(float(obj['area'])>=1024):
			num_1024 += 1

print("Objects larger than 128 account for {0}".format(num_128/num)) # 0.8919482248261609 --> 90
print("Objects larger than 256 account for {0}".format(num_256/num)) # 0.8123412677561173 --> 80
print("Objects larger than 512 account for {0}".format(num_512/num)) # 0.7111729984411399 --> 70
print("Objects larger than 1024 account for {0}".format(num_1024/num)) # 0.5950992240730036 --> 60


# Training Dataset
train_f = open(train_json,"r")
train_dict = json.load(train_f)
train_f.close()
_COCO = COCO(train_json)

num = 0
num_128 = 0
num_256 = 0
num_512 = 0
num_1024 = 0
for i, item in enumerate(train_dict["images"]):
	if(i % 1000==0):
		print(i)
	index_img = item["id"]
	annIds = _COCO.getAnnIds(imgIds = index_img, iscrowd = None)
	objs = _COCO.loadAnns(annIds)
	for obj in objs:
		num += 1
		if(float(obj['area'])>=128):
			num_128 += 1
		if(float(obj['area'])>=256):
			num_256 += 1
		if(float(obj['area'])>=512):
			num_512 += 1
		if(float(obj['area'])>=1024):
			num_1024 += 1

print(num, num_128, num_256, num_512, num_1024)
print("Objects larger than 128 account for {0}".format(num_128/num)) # 0.8896722477012322 --> 90
print("Objects larger than 256 account for {0}".format(num_256/num)) # 0.8063890321297528 --> 80
print("Objects larger than 512 account for {0}".format(num_512/num)) # 0.7010893318835902 --> 70
print("Objects larger than 1024 account for {0}".format(num_1024/num)) # 0.5849201311437925 --> 60
