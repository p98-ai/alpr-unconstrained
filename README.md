# ALPR in Unscontrained Scenarios

## Intro
The source code trains the license plate recognition and OCR model, and can output the test results of a single picture. However, the training of license plate recognition is implemented by Keras. Not only does the model take up too much GPU resources, but the inference speed ranks last in various frameworks. Here, TensorRT is used to deploy license plate recognition inference. The TensorRT tool launched by Nvidia to deploy models trained on mainstream frameworks can greatly increase the speed of model inference. It is often at least 1 times faster than the original framework, and it also consumes less memory . And because of the video stream access, for the output results, the use of time redundancy is for the same vehicle combined with multiple frames to fuse the results to improve the output accuracy.

## Prerequisit
TensorRT 7.0.0.11
numpy    1.14
onnx     1.6.0
opencv-python 4.2.0.32

## Usage
1、Build Darknet framework
2、Convert Keras model(json&h5) to onnx
```shellscript
$ python h52onnx.py
```  
3、Run inference.py(This includes LP detection and OCR inference process)
```shellscript
$ python inference.py --lpth 0.5 --ocrth 0.4 --onnx models/wpod0309b.onnx --e model0309b.engine --output samples/output --a test0305.mp4
```  

The following is to the source code and paper.


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

