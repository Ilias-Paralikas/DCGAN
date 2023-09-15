
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,noise_dimension=100,features_g=64,chanels=2):
        super(Generator,self).__init__()
        self.net  = nn.Sequential(
            self._block(noise_dimension,features_g*32,4,1,0),
            self._block(features_g*32,features_g*16,(1,4,4), (1,2,2), (0,1,1)),
            self._block(features_g*16,features_g*8,4,2,1),
            self._block(features_g*8,features_g*4,4,2,1),
            self._block(features_g*4,features_g*2,4,2,1),
            nn.ConvTranspose3d(features_g*2,chanels,4,2,1),
            nn.Tanh()
        )
        

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.LazyBatchNorm3d(out_channels),
            nn.ReLU(0.2)
        )
    
    def forward(self,x):
        return self.net(x)