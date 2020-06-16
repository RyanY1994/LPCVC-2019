# LPCVC-2019
2nd Prize in Object Detection Track of [2019 IEEE Low-Power Image Recognition Challenge (LPIRC) Online Track](https://lpcv.ai/competitions/2019) by Orange-Control, implemented by Tensorflow (TF>=1.12).


### Installation
The installation requirements are basically the same as that of the offical object-detection API of Tensorflow, please refer to [Installation.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

### Dataset Preparation
    1. Download [COCO-2017 dataset](http://cocodataset.org/#download) for Object Detection Task, in ./data/coco-2017.

    2. Run genMinval8000.py in ./data/coco/coco-ovic to generate annotation files (i.e., ./annotations/instances_train_L.json and ./annotations/instances_minval_L.json) for training and validataion according to ovic_val_2017_list.txt required by [OVIC](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt).

    3. Run splitImage.py in ./data/coco/coco-ovic to split images according to ovic_val_2017_list.txt. The splitted images are stored in ./data/coco/coco-ovic/val_ovic and ./data/coco/coco-ovic/train_ovic.

    4. Run removeSmallObj.py in ./data/coco/coco-ovic to generate annotation file (i.e., ./annotations/instances_train_L_256.json) for images with area larger or equal to 256 for the removal of small objects.

### Training
##### Stage1: Load pretrained model ([ssd_mobilenet_v2_quantized_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz)) and train for 200K with lr 0.0004 with input size 320x320 --> AP 20.7

    CUDA_VISIBLE_DEVICES=0 python ./LPCVC-2019/model_main.py \
        --pipeline_config_path=./LPCVC-2019/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco_check_9.config \
        --model_dir=./LPCVC-2019/tmp/model_trained/lpcvc4_check_9/ \
        --num_train_steps=200000 \
        --sample_1_of_n_eval_examples=8 \
        --alsologtostderr

##### Stage2: Load model from Stage1 and train for 70K with lr 0.00005 --> AP 21.1

    CUDA_VISIBLE_DEVICES=0 python ./LPCVC-2019/model_main.py \
        --pipeline_config_path=./LPCVC-2019/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco_check_19.config \
        --model_dir=./LPCVC-2019/tmp/model_trained/lpcvc4_check_19/ \
        --num_train_steps=1000000 \
        --sample_1_of_n_eval_examples=8 \
        --alsologtostderr
  
##### Stage3: Load model from Stage2, change min_scale of anchor_generator to 0.15, and train with instances_train_L_256.json. Train for 150K with lr 0.0004 (AP 21.1), for 150K with lr 0.00005 (AP 21.6), for 250K with lr 0.00001 --> AP 21.7

    CUDA_VISIBLE_DEVICES=0 python ./LPCVC-2019/model_main.py \
        --pipeline_config_path=./LPCVC-2019/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco_check_43.config \
        --model_dir=./LPCVC-2019/tmp/model_trained/lpcvc4_check_43/ \
        --num_train_steps=1000000 \
        --sample_1_of_n_eval_examples=8 \
        --alsologtostderr



### Testing
##### Convert .ckpt (refer to the [link](https://drive.google.com/open?id=1DSXwwqAxG003ja3-Lr_KAwjvh9LOKkN5)) to .pb for tflite

    python ./LPCVC-2019/export_tflite_ssd_graph.py \
        --pipeline_config_path=./LPCVC-2019/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco_check_43.config \
        --trained_checkpoint_prefix=./LPCVC-2019/tmp/model_trained/lpcvc4_check_43/model.ckpt-560197 \
        --output_directory=./LPCVC-2019/tmp/model_trained/lpcvc4_check_43/tflite/560197/non_34_convert/ \
        --max_detections=100 \
        --add_postprocessing_op=true
 
##### Convert .pb to .tflite (refer to the [link](https://drive.google.com/open?id=1DSXwwqAxG003ja3-Lr_KAwjvh9LOKkN5))

    tflite_convert --graph_def_file=./LPCVC-2019/tmp/model_trained/lpcvc4_check_43/tflite/560197/non_34_convert/tflite_graph.pb \
    --output_format=TFLITE \
    --output_file=./LPCVC-2019/tmp/model_trained/lpcvc4_check_43/tflite/560197/non_34_convert/model.tflite \
    --input_shapes=1,320,320,3 \
    --input_arrays=normalized_input_image_tensor \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_dev_values=128 \
    --allow_custom_ops

##### Calculate AP of tflite locally 

    Run tflite_test.py in ./data/coco/coco-ovic to test on 7991 images in ./data/coco/coco-ovic/val_ovic.

### Trained Models
You can download our trained models (.ckpt and .tflite) through the [link](https://drive.google.com/open?id=1DSXwwqAxG003ja3-Lr_KAwjvh9LOKkN5). Note that the three models produce similar result on AP metric.




