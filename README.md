# ALPR in Unscontrained Scenarios

##1227更新
本版本目前舍弃了vehicle-detection这步，因为所用测试集中几乎所有图片只含一辆车，且车头或车尾占整张图片比重较大。变化主要有：
1、输出时去掉对车辆画黄框这步，直接在整张图片中按比例用红线圈出车牌
2、针对中文车牌，因cv2无法输出中文字符，改用PIL输出中文字符
3、针对双层车牌，目前采用计算车牌识别狂左上角高度坐标的均值和方差进行；显然单层车牌该项方差应绝对小于双层车牌（经对图像质量较好的车牌测试，单层车牌该项方差在0.0001这个级别，而双层车牌该项方差在0.01这个级别）所以以0.001为方差阈值来划分单双层；而对每张双层车牌判断字符的上下层归属则采用：字符识别狂左上角高度坐标小于均值的属于上层，大于均值的属于下层；该解决方案对质量好的车牌图片识别率很高，但由于车牌图片产生于前一步的license-plate-detection，准确率不太高，会出现误识别或框出了车牌但角度奇怪的情况，对于这些情况，不太很好解决

## Introduction

This repository contains the author's implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

* Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

If you use results produced by our code in any publication, please cite our paper:

```
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}
```

## Requirements

In order to easily run the code, you must have installed the Keras framework with TensorFlow backend. The Darknet framework is self-contained in the "darknet" folder and must be compiled before running the tests. To build Darknet just type "make" in "darknet" folder:

```shellscript
$ cd darknet && make
```

**The current version was tested in an Ubuntu 16.04 machine, with Keras 2.2.4, TensorFlow 1.5.0, OpenCV 2.4.9, NumPy 1.14 and Python 2.7.**

## Download Models

After building the Darknet framework, you must execute the "get-networks.sh" script. This will download all the trained models:

```shellscript
$ bash get-networks.sh
```

## Running a simple test

Use the script "run.sh" to run our ALPR approach. It requires 3 arguments:
* __Input directory (-i):__ should contain at least 1 image in JPG or PNG format;
* __Output directory (-o):__ during the recognition process, many temporary files will be generated inside this directory and erased in the end. The remaining files will be related to the automatic annotated image;
* __CSV file (-c):__ specify an output CSV file.

```shellscript
$ bash get-networks.sh && bash run.sh -i samples/test -o /tmp/output -c /tmp/output/results.csv
```

## Training the LP detector

To train the LP detector network from scratch, or fine-tuning it for new samples, you can use the train-detector.py script. In folder samples/train-detector there are 3 annotated samples which are used just for demonstration purposes. To correctly reproduce our experiments, this folder must be filled with all the annotations provided in the training set, and their respective images transferred from the original datasets.

The following command can be used to train the network from scratch considering the data inside the train-detector folder:

```shellscript
$ mkdir models
$ python create-model.py eccv models/eccv-model-scracth
$ python train-detector.py --model models/eccv-model-scracth --name my-trained-model --train-dir samples/train-detector --output-dir models/my-trained-model/ -op Adam -lr .001 -its 300000 -bs 64
```

For fine-tunning, use your model with --model option.

## A word on GPU and CPU

We know that not everyone has an NVIDIA card available, and sometimes it is cumbersome to properly configure CUDA. Thus, we opted to set the Darknet makefile to use CPU as default instead of GPU to favor an easy execution for most people instead of a fast performance. Therefore, the vehicle detection and OCR will be pretty slow. If you want to accelerate them, please edit the Darknet makefile variables to use GPU.
