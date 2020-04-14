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
