import cv2
import os
import numpy as np
from os.path import splitext
import torch
from torch.utils.data import Dataset 
from src.utils import image_files_from_folder
from src.label import readShapes
from src.sampler import augment_sample, labels2output_map

def process_data_item(data_item,dim,model_stride):
	XX,llp,pts = augment_sample(data_item[0],data_item[1].pts,dim)
	YY = labels2output_map(llp,pts,dim,model_stride)
	return XX,YY

class LPDataset(Dataset):  # for training/testing
    def __init__(self, path, img_size=208):
        self.img_files = image_files_from_folder(path)

        n = len(self.img_files)
        assert n > 0, 'No images found in %s' % path
        self.img_size = img_size
#         self.label_files = labels_files_from_folder(path)
        # if n < 200:  # preload all images into memory if possible
        #    self.imgs = [cv2.imread(img_files[i]) for i in range(n)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
#         label_path = self.label_files[index]
        model_stride = 2**4
        labfile = splitext(img_path)[0] + '.txt'
        L = readShapes(labfile)
        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'File Not Found ' + img_path
        data = [img,L[0]]
        img,labels = process_data_item(data,self.img_size,model_stride)
        
        
#        labels = []
#        if os.path.isfile(label_path):
#            with open(label_path, 'r') as file:
#                for line in file:
#                    data = line.strip().split(',')
#            labels = data[1:9]
#            text = data[9]
        # Normalize
        img = img[:, :, ::-1]- np.zeros_like(img,dtype=np.float32)
        img = img.transpose(2, 0, 1)
        # BGR to RGB, to 3x416x416
#        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
#        img /= 255.0  # 0 - 255 to 0.0 - 1.0
#        labels = np.array(labels).reshape(2,4)
        labels = np.ascontiguousarray(labels, dtype=np.float32)
        labels = torch.from_numpy(labels)
#        labels = labels.requires_grad_()
        img = torch.from_numpy(img)
#        img = img.requires_grad_()
        return img , labels    #, (h, w)



