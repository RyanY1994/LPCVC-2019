import os 
import shutil
import numpy as np
#import cv2
import json
import sys 

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask



"""
out_json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
"""
# ann_root = "/nfs/project/OVIC/coco_2014_2017/"
# images_root = "/nfs/project/OVIC/coco_ovic/"
coco_images_root = "./../coco-2017/"
ovic_images_root = "./"
now_val_image_split = os.path.join(coco_images_root, "val2017")
now_train_image_split = os.path.join(coco_images_root, "train2017")

out_val_image_split = os.path.join(ovic_images_root, "val_ovic")
out_train_image_split = os.path.join(ovic_images_root, "train_ovic")

minval_txt_f = open(os.path.join("ovic_val_2017_list.txt"), "r")

image_ids = []
for line in minval_txt_f.readlines():
    temp = line.strip()
    _,fname = os.path.split(temp)
    image_ids.append(fname)
    
c = 0    

for root,dirs,files in os.walk(now_train_image_split):
    for file in files:
        if(not file.lower().endswith(".jpg")): continue
        src_image = os.path.join(root,file)
        if(file in image_ids):
            c += 1
            dst_image = os.path.join(out_val_image_split,file)
        else:
            dst_image = os.path.join(out_train_image_split,file)
        if(c%100==0):
            print(c)
        #print(dst_image)
        shutil.move(src_image,dst_image)
        


for root,dirs,files in os.walk(now_val_image_split):
    for file in files:
        if(not file.lower().endswith(".jpg")): continue
        src_image = os.path.join(root,file)
        if(file in image_ids):
            c += 1
            dst_image = os.path.join(out_val_image_split,file)
            print(c)
        else:
            dst_image = os.path.join(out_train_image_split,file)
        #print(dst_image)
        shutil.move(src_image,dst_image)
