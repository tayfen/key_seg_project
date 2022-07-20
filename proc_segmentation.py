import os
import torchvision.transforms as tt
import torch.nn.functional as F
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
from torch import Tensor
import torch.nn as nn
import numpy as np
import logging
import random
from PIL import Image

# в ходе экспериментов пробовал различные размеры, начинал с маленьких, но кажется, что чем больше информации - тем лучше, так что лучше не сжимать
# аналогичная история с каналами, начинал с 1 канальных изображений, но решил оставить 3 канальные входные
RESCALE_SIZE = 512 #попробовать 1024
CROP = 20

def get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return TF.pad(image, padding, 0, 'constant')
  
class RandomRotation(tt.RandomRotation):
    def __init__(self, *args, **kwargs):
        super(RandomRotation, self).__init__(*args, **kwargs) # let super do all the work

        self.angle = self.get_params(self.degrees) # initialize your random parameters

    def forward(self, img): # override T.RandomRotation's forward
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        return TF.rotate(img, self.angle, self.resample, self.expand, self.center, fill)

class FilteredKeySegDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):   # initial logic happens like transform

         self.image_paths = image_paths
         self.target_paths = target_paths
            
    def transform(self, image, mask):
        
        img_w, img_h = image.size
        # Random crop
        i, j, h, w = tt.RandomCrop.get_params(
            image, output_size=(img_h - CROP, img_w - CROP))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        common_transform = tt.Compose([
                    SquarePad(),
                    tt.ToTensor(),            
                    tt.Resize([RESCALE_SIZE, RESCALE_SIZE]),
                    #tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    #tt.Grayscale(num_output_channels=1),
                    #tt.RandAugment(),
                    #tt.RandomHorizontalFlip(p=0.5),
                ])
        
        image_only_transform = tt.Compose([
                    #tt.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2)),
                    tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    tt.RandomAdjustSharpness(sharpness_factor=2),
                    #tt.RandomAutocontrast(),
                    #tt.RandomEqualize(),
        ]) 
        
        mask_only_transform = tt.Compose([
                    tt.Grayscale(num_output_channels=1),
        ])

        #image = image_only_transform(image)
        mask = mask_only_transform(mask)
        
        image = common_transform(image)
        mask = common_transform(mask)
        
        #image = image_only_transform(image)
        
        mask = mask > 0
        #mask = mask + 1
        #mask = mask - mask.min()
        #mask = mask / (mask.max() - mask.min())

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        rotate = RandomRotation(degrees=10)
        image = rotate(image)
        mask = rotate(mask)

        #image = image_only_transform(image)
        # Transform to tensor
        #image = TF.to_tensor(image)
        #mask = TF.to_tensor(mask)        

        return image, mask

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        t_image, t_mask = self.transform(image, mask)
        return t_image, t_mask

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)

class UNet(nn.Module):
    batchNorm_momentum = 0.1
    def __init__(self):
        super().__init__()

        self.enc_conv0 = self.make_enc_layer(3, 64)
        self.enc_conv1 = self.make_enc_layer(64, 128)
        self.enc_conv2 = self.make_enc_layer(128, 256)
        self.enc_conv3 = self.make_enc_layer(256, 512)

        self.bridge = nn.Sequential(
                nn.Conv2d(512,1024,kernel_size = 3, padding = 1),
                nn.BatchNorm2d(1024, momentum= self.batchNorm_momentum),
                nn.ReLU(),
                nn.Conv2d(1024,1024,kernel_size = 3, padding = 1),
                nn.BatchNorm2d(1024, momentum= self.batchNorm_momentum),
                nn.ReLU(),
            )
        
        self.dec_conv0 = self.make_dec_layer(1024 + 512, 512)
        self.dec_conv1 = self.make_dec_layer(512 + 256, 256)
        self.dec_conv2 = self.make_dec_layer(256 + 128, 128)
        self.dec_conv3 = self.make_dec_layer(128 + 64, 64)

        self.exit = nn.Conv2d(64,1,kernel_size = 1, padding = 0)

        self.upsample = nn.Upsample(scale_factor=2)

    def make_enc_layer(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input,output,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output, momentum= self.batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(output,output,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output, momentum= self.batchNorm_momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
    def make_dec_layer(self, input, output):
        return nn.Sequential(
            nn.Conv2d(input,output,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output, momentum= self.batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(output,output,kernel_size = 3, padding = 1),
            nn.BatchNorm2d(output, momentum= self.batchNorm_momentum),
            nn.ReLU(),
        )        

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        
        bridge = self.bridge(e3)
        
        # decoder
        d0 = self.dec_conv0(self.upsample(torch.cat([bridge, e3], dim=1)))
        d1 = self.dec_conv1(self.upsample(torch.cat([d0, e2], dim=1)))
        d2 = self.dec_conv2(self.upsample(torch.cat([d1, e1], dim=1)))
        d3 = self.dec_conv3(self.upsample(torch.cat([d2, e0], dim=1)))
        
        return self.exit(d3)