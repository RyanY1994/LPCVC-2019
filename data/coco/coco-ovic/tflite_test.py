import numpy as np
import tensorflow as tf
import cv2 
import os 
from coco import COCODetection
import pickle


COCOroot = "/data2/chengxiang/models/research/object_detection/data/coco/coco-ovic/"


def evalImage_fix(model_path, save_path, num_samples):
	
	# Load TFLite model and allocate tensors
	interpreter = tf.contrib.lite.Interpreter(model_path)
	interpreter.allocate_tensors()

	# Get input and output tensors
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	all_boxes = [[np.array([]) for _ in range(num_samples)] for _ in range(80)]
	
	class_90_80_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}

	testset = COCODetection(COCOroot, ['instances_minval_L'], None)
	
	for i in range(num_samples):
		#if(i>10): break
		print(i)
		image = testset.pull_image(i)
		init_shape = image.shape
		Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		Image = cv2.resize(Image, (320, 320))
		input_image = np.array([Image], dtype = np.uint8)
		pred = tfliteDetect(init_shape, input_image, interpreter, input_details, output_details)
		
		for j in range(80):
			temp = [x[0:5] for x in pred if class_90_80_map[int(x[-1])+1] == j]
			all_boxes[j][i] = np.array(temp)
	
	print('Evaluating detections')
	testset.evaluate_detections(np.array(all_boxes), save_path)

def tfliteDetect(init_shape, Image, interpreter, input_details, output_details):

	# Test model on random input data.
	input_shape = input_details[0]['shape']
	interpreter.set_tensor(input_details[0]['index'], Image)
	interpreter.invoke()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	re = []
	bbox = interpreter.get_tensor(output_details[0]['index'])[0]
	
	b_cls = interpreter.get_tensor(output_details[1]['index'])[0]
	b_score =  interpreter.get_tensor(output_details[2]['index'])[0]
	
	for i in range(100):
		bbox[i][0] *= init_shape[0]
		bbox[i][2] *= init_shape[0]
		bbox[i][1] *= init_shape[1]
		bbox[i][3] *= init_shape[1]
		bbox[i] = [bbox[i][1],bbox[i][0],bbox[i][3],bbox[i][2]]
		temp = list(bbox[i]) + list([b_score[i]]) + list([b_cls[i]])
		re.append(temp)	
	return re

# model_path = "/data2/chengxiang/models/research/object_detection/tmp/model_trained/lpcvc4_check_19/tflite/70317/non_34_convert/model_17.tflite"
# model_path = "/data2/chengxiang/models/research/object_detection/tmp/model_trained/lpcvc4_check_20/tflite/69919/non_34_convert/model_16.tflite"
model_path = "/data2/chengxiang/models/research/object_detection/data/coco/coco-ovic/LPCVC/model_83.tflite"
save_path = "/data2/chengxiang/models/research/object_detection/data/coco/coco-ovic/test_2/"
num_samples = 7991
evalImage_fix(model_path, save_path, num_samples)
