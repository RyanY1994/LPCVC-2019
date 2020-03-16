import os 
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
# out_root = "/nfs/project/OVIC/coco_ovic/"
ann_root = "./../coco-2017/"
out_root = "./"
train_json = os.path.join(ann_root, "annotations/instances_train2017.json")
val_json = os.path.join(ann_root, "annotations/instances_val2017.json")
out_minval_json = os.path.join(out_root, "annotations/instances_minval_L.json")
out_train_json = os.path.join(out_root, "annotations/instances_train_L.json")

out_minval_json_dict = {"images":[], "type": "instances", "annotations": [], "categories": []}
out_train_json_dict = {"images":[], "type": "instances", "annotations": [], "categories": []}

minval_txt_f = open(os.path.join("ovic_val_2017_list.txt"), "r")

image_ids = []
for line in minval_txt_f.readlines():
	temp = line.strip()
	_,fname = os.path.split(temp)
	image_ids.append(fname)
	
c = 0    
	 
val_f = open(val_json,"r")
val_dict = json.load(val_f)
val_f.close()
_COCO = COCO(val_json)
for i,item in enumerate(val_dict["images"]): 
	#if(c>2): break
	img_id = item["file_name"]
	index_img = item["id"]
	if(img_id in image_ids):
		c += 1
		out_minval_json_dict["images"].append(item)
		annIds = _COCO.getAnnIds(imgIds=index_img, iscrowd=None)
		objs = _COCO.loadAnns(annIds)
		#print(objs)
		out_minval_json_dict['annotations'].extend(objs)
	else:
		out_train_json_dict["images"].append(item)
		annIds = _COCO.getAnnIds(imgIds=index_img, iscrowd=None)
		objs = _COCO.loadAnns(annIds)
		#print(objs)
		out_train_json_dict['annotations'].extend(objs)

print(c)


train_f = open(train_json,"r")
train_dict = json.load(train_f)
train_f.close()
_COCO = COCO(train_json)
for i,item in enumerate(train_dict["images"]): 
	#if(c>3): break
	img_id = item["file_name"]
	index_img = item["id"]
	if(img_id in image_ids):
		c += 1
		out_minval_json_dict["images"].append(item)
		annIds = _COCO.getAnnIds(imgIds=index_img, iscrowd=None)
		objs = _COCO.loadAnns(annIds)
		out_minval_json_dict['annotations'].extend(objs)
	else:
		out_train_json_dict["images"].append(item)
		annIds = _COCO.getAnnIds(imgIds=index_img, iscrowd=None)
		objs = _COCO.loadAnns(annIds)
		out_train_json_dict['annotations'].extend(objs)
	if(c%1000==0):
		print(c)        
		
		
out_minval_json_dict["categories"] = val_dict["categories"]
out_train_json_dict["categories"] = val_dict["categories"]   

#print(c,out_minval_json_dict["annotations"][-4:-1])    

with open(out_minval_json,"w") as f:
	json.dump(out_minval_json_dict,f)
	print("write minval end!")
	
with open(out_train_json,"w") as f:
	json.dump(out_train_json_dict,f)
	print("write train end!")    
	
