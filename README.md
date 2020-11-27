# yolov4 pytorch in my program

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [x] Train
    - [x] Mocaic


## 项目中用来训练闸机的行人检测以及行人跟踪

  个人代码注释以及解读说明，使用的方法和函数
# 0. Weights Download

## 0.1 darknet
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch
you can use darknet2pytorch to convert it yourself, or download my converted model.


# 1. Train

[use yolov4 to train your own data](Use_yolov4_to_train_your_own_data.md)
for training darknet version see https://github.com/AlexeyAB/darknet

1. Download weight
2. Transform data

    For coco dataset,you can use tool/coco_annotation.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
    change  get_image_id function  in dataset.py for evaluating 

3. Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. Inference

## 2.1 Performance on MS COCO dataset (using pretrained DarknetWeights from <https://github.com/AlexeyAB/darknet>)




## 2.3 **Different inference options**

- Load the pretrained darknet model and darknet weights to do the inference (image size is configured in cfg file already)

    ```sh
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
    ```

- Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
    
- Load converted ONNX file to do inference (See section 3 and 4)

- Load converted TensorRT engine file to do inference (See section 5)

## 2.4 Inference output

There are 2 inference outputs.
- One is locations of bounding boxes, its shape is  `[batch, num_boxes, 1, 4]` which represents x1, y1, x2, y2 of each bounding box.
- The other one is scores of bounding boxes which is of shape `[batch, num_boxes, num_classes]` indicating scores of all classes for each bounding box.

Until now, still a small piece of post-processing including NMS is required. We are trying to minimize time and complexity of post-processing.



## 2.5 Run the demo

```sh
python demo_trt.py <tensorRT_engine_file> <input_image> <input_H> <input_W>
```

- This demo here only works when batchSize is dynamic (1 should be within dynamic range) or batchSize=1, but you can update this demo a little for other dynamic or static batch sizes.
    
- Note1: input_H and input_W should agree with the input size in the original ONNX file.
    
- Note2: extra NMS operations are needed for the tensorRT output. This demo uses python NMS code from `tool/utils.py`.



Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3
- https://github.com/Tianxiaomo/pytorch-YOLOv4

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
