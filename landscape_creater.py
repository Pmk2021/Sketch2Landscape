import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
from PIL import Image
from random import randint
from scipy import ndimage
from skimage.transform import resize
import urllib.request
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature, filters
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from random import randint
from copy import deepcopy
from PIL import ImageFile

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.dropout = nn.Dropout2d(0.5)
        #Downsampling
        self.conv1 = nn.Conv2d(4,64, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv1.weight,mean=0, std=0.02)
        self.conv2 = nn.Conv2d(64,128, 4, stride=2, padding =  1,bias=False)
        nn.init.normal_(self.conv2.weight,mean=0, std=0.02)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv3.weight,mean=0, std=0.02)
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv4.weight,mean=0, std=0.02)
        self.norm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv5.weight,mean=0, std=0.02)
        self.norm5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512,512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv6.weight,mean=0, std=0.02)
        self.norm6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512,512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv7.weight,mean=0, std=0.02)
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv8.weight,mean=0, std=0.02)
        self.norm8 = nn.BatchNorm2d(512)
        #Upsampling
        self.conv9 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv9.weight, mean=0, std=0.02)
        self.norm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv10.weight,mean=0, std=0.02)
        self.norm10 = nn.BatchNorm2d(512)
        self.conv11 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv11.weight,mean=0, std=0.02)
        self.norm11 = nn.BatchNorm2d(512)
        self.conv12 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv12.weight,mean=0, std=0.02)
        self.norm12 = nn.BatchNorm2d(512)
        self.conv13 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv13.weight,mean=0, std=0.02)
        self.norm13 = nn.BatchNorm2d(256)
        self.conv14 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv14.weight,mean=0, std=0.02)
        self.norm14 = nn.BatchNorm2d(128)
        self.conv15 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding = 1,bias=False)
        nn.init.normal_(self.conv15.weight,mean=0, std=0.02)
        self.norm15 = nn.BatchNorm2d(64)
        self.conv16 = nn.ConvTranspose2d(128, 3, 4, stride=2, padding = 1)
        nn.init.normal_(self.conv1.weight,mean=0, std=0.02)
    def forward(self, x):
      #Downsampling
      s1 = F.leaky_relu(self.conv1(x))
      s1 = self.dropout(s1)
      s2 = F.leaky_relu(self.norm2(self.conv2(s1)))
      s2 = self.dropout(s2)
      s3 = F.leaky_relu(self.norm3(self.conv3(s2)))
      s3 = self.dropout(s3)
      s4 = F.leaky_relu(self.norm4(self.conv4(s3)))
      s5 = F.leaky_relu(self.norm5(self.conv5(s4)))
      s6 = F.leaky_relu(self.norm6(self.conv6(s5)))
      s7 = F.leaky_relu(self.norm7(self.conv7(s6)))
      s8 = F.leaky_relu(self.conv8(s7))

      #Upsampling
      s7_1 = F.relu(self.norm9(self.conv9(s8)))
      s6_1 = F.relu(self.norm10(self.conv10(torch.cat((s7,s7_1),1))))
      s5_1 = F.relu(self.norm11(self.conv11(torch.cat((s6,s6_1),1))))
      s4_1 = F.relu(self.norm12(self.conv12(torch.cat((s5,s5_1),1))))
      s3_1 = F.relu(self.norm13(self.conv13(torch.cat((s4,s4_1),1))))
      s3_1 = self.dropout(s3_1)
      s2_1 = F.relu(self.norm14(self.conv14(torch.cat((s3,s3_1),1))))
      s2_1 = self.dropout(s2_1)
      s1_1 = F.relu(self.norm15(self.conv15(torch.cat((s2,s2_1),1))))
      s1_1 = self.dropout(s1_1)
      s0 = F.tanh(self.conv16(torch.cat((s1,s1_1),1)))
      return s0
  

Generator = Net1()
Generator.load_state_dict(torch.load('landscape_generator',map_location=torch.device('cpu')))
Generator.eval()
