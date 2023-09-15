
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

class MyDataLoader(Dataset):
    def __init__(self,
                 resized_folder= os.path.join(os.pardir, 'sliced_data'),
                 dimensions =128):
                 
        self.dimensions = dimensions
        self.resized_folder  = resized_folder
        self.filenames = [os.path.join(self.resized_folder,f) for f in os.listdir(self.resized_folder)]
        self.len = len(self.filenames)
        
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        return torch.load(self.filenames[idx]),1
    
    def visualize(self,idx=0,outsider=None):
      if outsider ==None:
          patient = self.__getitem__(idx)[0]
      else:
          patient = outsider
      medical_img =  patient[0]
      mask = patient[1]
      mask = torch.ceil(mask)
      masked = np.ma.masked_where(mask == 0, mask)

      w = 30
      h = 30
      fig = plt.figure(figsize=(20, 20))
      columns = 8
      rows = 8
      for i in range(0, columns*rows):
          img = medical_img[i,:,:]
          fig.add_subplot(rows, columns, i+1)
          plt.imshow(img, cmap='gray')
          plt.imshow(masked[i,:,:],'prism', interpolation='none', alpha=1)
      plt.show()
      