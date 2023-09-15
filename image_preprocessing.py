
import os
from monai.transforms import LoadImage,Orientation
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from glob import glob
import torchvision.utils as vutils
# import nrrd
from torch.utils.data import Dataset
import random

class ImagePreprocessor():
    def __init__(self,
                 img_dir='Acrin_data',
                 mask_dir = 'resampled_masks'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_numbers =[]
        for img in os.listdir(self.img_dir):
          img_num  =  "".join([ele for ele in img if ele.isdigit()])
          self.file_numbers.append(img_num)


    # def health_check(self):
    #   assert len(self.img_names) ==len(self.mask_names)
    #   for i in range(len(self.img_names)):
    #     img_num  = "".join([ele for ele in self.img_names[i] if ele.isdigit()])
    #     mask_num  = "".join([ele for ele in  self.mask_names[i] if ele.isdigit()])
    #     assert img_num==mask_num
    #     img_name = os.path.join(self.img_dir,self.img_names[i])
    #     img = LoadImage(image_only=True,ensure_channel_first=True, simple_keys=True)(img_name)
    #     img_shape = img.shape
    #     mask_name= os.path.join(self.mask_dir,self.mask_names[i])
    #     mask = nrrd.read(mask_name)
    #     mask_shape =mask[0].shape
    #     assert img_shape[1:] == mask_shape
    #     print(mask_shape)

    def get_img_filename(self,idx):
      return os.path.join(self.img_dir,'patient'+str(idx)+'.nii.gz')
    def get_mask_filename(self,idx):
      return os.path.join(self.mask_dir,'segmentation'+str(idx)+'.nrrd')



    def get_min_slices(self):
      self.mask_coordinates = {}
      min_size_scan = 1000
      max_non_zeromask_length =0
      for file_number in self.file_numbers:
        mask_file = self.get_mask_filename(file_number)
        #mask  =nrrd.read(mask_file)[0]
        mask = LoadImage(reader="itkreader")(mask_file)[0]
        min_size_scan = min(min_size_scan,mask.shape[2])
        coordinate = [-1,-1]
        for i in range(mask.shape[2]):
            if np.any(mask[:,:,i] !=0):
                if coordinate[0]==-1:
                    coordinate[0]= i
                coordinate[1]=i

        self.mask_coordinates[str(file_number)] = coordinate
        max_non_zeromask_length = max((coordinate[1]-coordinate[0]+1),max_non_zeromask_length)


      self.slices_size = min(max_non_zeromask_length,min_size_scan)

    def slice_images(self,
                    sliced_data_dir='sliced_data',
                    percent_offset=0.3,
                    dimensions=128):
      if not os.path.exists(sliced_data_dir):
        os.mkdir(sliced_data_dir)

      self.get_min_slices()

      for i in self.file_numbers:
        print(i)
        img_file = self.get_img_filename(i)
        mask_file = self.get_mask_filename(i)
        # mask  =nrrd.read(mask_file)[0]
        mask = LoadImage(reader="itkreader")(mask_file)[0]
        img = LoadImage(image_only=True)(img_file)
        while self.mask_coordinates[i][1] - self.mask_coordinates[i][0] > self.slices_size:
          self.mask_coordinates[i][0]  +=1
          self.mask_coordinates[i][1]  -=1



        slices_start_max = min(self.mask_coordinates[i][0],  mask.shape[2]  -self.slices_size)
        slices_start_min = max(self.mask_coordinates[i][1] - self.slices_size,0)
        # print('slices_start_min: ',slices_start_min)
        # print('slices_start_max: ',slices_start_max)
        start =random.randint(slices_start_min,slices_start_max)
        # print("mask dimension : ", mask.shape[2])
        # print(self.mask_coordinates[i])
        # # print(start)
        # print('#####################')
        mask =  Orientation(axcodes='RAS')(mask)
        mask = T.Resize(64)(mask)
        mask = mask.permute(2,0,1)
        total_offset = int(percent_offset  * mask.shape[2])
        mask = mask[:,:,total_offset:]

        img =  Orientation(axcodes='RAS')(img)
        img = T.Resize(64)(img)
        img = img.permute(2,0,1)
        total_offset = int(percent_offset  * img.shape[2])
        img = img[:,:,total_offset:]
        #img = T.Normalize((torch.mean(img)),(torch.max(img)))(img)
        #danger

        patient = torch.stack((img,mask))
        patient = T.Resize((dimensions,dimensions))(patient)
        print(patient.shape)
        filename = os.path.join(sliced_data_dir,'patient'+str(i) +'.pt')
        torch.save(patient,filename)

Preprocessor = ImagePreprocessor()
Preprocessor.slice_images()