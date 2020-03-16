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
train_json_128_f = os.path.join(out_root, "annotations/instances_train_L_128.json")
train_json_256_f = os.path.join(out_root, "annotations/instances_train_L_256.json")
train_json_512_f = os.path.join(out_root, "annotations/instances_train_L_512.json")
train_json_1024_f = os.path.join(out_root, "annotations/instances_train_L_1024.json")


train_json_128 =  {"images":[], "type": "instances", "annotations": [],
				 "categories": []}
train_json_256 =  {"images":[], "type": "instances", "annotations": [],
				 "categories": []}
train_json_512 =  {"images":[], "type": "instances", "annotations": [],
				 "categories": []}
train_json_1024 = {"images":[], "type": "instances", "annotations": [],
				 "categories": []}

train_f = open(train_json,"r")
train_dict = json.load(train_f)
train_f.close()
_COCO = COCO(train_json)

for i,item in enumerate(train_dict["images"]): 
	if(i % 1000==0):
		print(i)
	index_img = item["id"]
	annIds = _COCO.getAnnIds(imgIds = index_img, iscrowd = None)
	objs = _COCO.loadAnns(annIds)

	for obj in objs:
		if(float(obj['area'])>=128):
			train_json_128['annotations'].append(obj)
		if(float(obj['area'])>=256):
			train_json_256['annotations'].append(obj)
		if(float(obj['area'])>=512):
			train_json_512['annotations'].append(obj)
		if(float(obj['area'])>=1024):
			train_json_1024['annotations'].append(obj)     
		
train_json_128["images"] = train_dict["images"]  
train_json_128["categories"] = train_dict["categories"] 
train_json_256["images"] = train_dict["images"]  
train_json_256["categories"] = train_dict["categories"] 
train_json_512["images"] = train_dict["images"]  
train_json_512["categories"] = train_dict["categories"] 
train_json_1024["images"] = train_dict["images"]  
train_json_1024["categories"] = train_dict["categories"]  

with open(train_json_128_f,"w") as f:
	json.dump(train_json_128,f)
	print("write train_json_128 end!") 
with open(train_json_256_f,"w") as f:
	json.dump(train_json_256,f)
	print("write train_json_256 end!") 
with open(train_json_512_f,"w") as f:
	json.dump(train_json_512,f)
	print("write train_json_512 end!") 
with open(train_json_1024_f,"w") as f:
	json.dump(train_json_1024,f)
	print("write train_json_1024 end!")    
	

