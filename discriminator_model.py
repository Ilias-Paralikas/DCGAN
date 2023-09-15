import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,features_d=32, in_channels=2):
        super(Discriminator,self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels,features_d,(1,4,4), (1,2,2), (0,1,1),bias=False),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            self._block(features_d*8,features_d*16,4,2,1),
            nn.Conv3d(features_d*16,1,4,2,0),
            nn.Sigmoid()


        )
        
    def _block(self,in_channels,out_channels,kernel_size,stride,padding,p=0.5):
        return nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.Dropout3d(p=p),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.net(x)
