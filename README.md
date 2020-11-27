# yolov4 pytorch in my program

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

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



## 2.3 **Different inference options**

- Load the pretrained darknet model and darknet weights to do the inference (image size is configured in cfg file already)

    ```sh
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
    ```

- Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```




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
