import torch, kornia as K, kornia.augmentation as KA
from common import *
torch.manual_seed(0)
img=load('astro.png')
box=torch.tensor([[[170.,20.],[290.,20.],[290.,170.],[170.,170.]]])  # face, xy corners
aug=KA.AugmentationSequential(
  KA.RandomHorizontalFlip(p=0.5),
  KA.RandomAffine(degrees=25,translate=(0.1,0.1),scale=(0.8,1.1),p=1.0),
  KA.ColorJitter(0.3,0.3,0.3,0.05,p=0.8),
  data_keys=['input','bbox'])
outs=[];labs=[]
batch=img.repeat(8,1,1,1); bb=box[None].repeat(8,1,1,1)
o,b=aug(batch,bb)
print('out',tuple(o.shape),'boxes',tuple(b.shape))
import cv2,numpy as np
for i in range(8):
    x=(o[i].permute(1,2,0).numpy()*255).astype('uint8').copy()
    pts=b[i,0].numpy().astype(int)
    cv2.polylines(x,[pts.reshape(-1,1,2)],True,(255,40,40),3)
    outs.append(torch.from_numpy(x).permute(2,0,1)[None].float()/255); labs.append(f'样本 {i+1}')
grid(outs,labs,'images/03-augment.png')
print('box[0] after aug:',b[0,0].round().tolist())
