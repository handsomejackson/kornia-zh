import torch, kornia as K
from common import *
img=load('astro.png'); print('shape',tuple(img.shape),img.dtype,'range',float(img.min()),float(img.max()))
gray=K.color.rgb_to_grayscale(img)
blur=K.filters.gaussian_blur2d(img,(9,9),(3.,3.))
sob=K.filters.sobel(gray)
mag,edges=K.filters.canny(gray)
hsv=K.color.rgb_to_hsv(img); hsv[:,0]=(hsv[:,0]+2.0)%(2*3.14159); hue=K.color.hsv_to_rgb(hsv)
sharp=K.enhance.sharpness(img,2.0)
eq=K.enhance.equalize(img)
grid([img,gray,blur,sob*3,edges,hue,sharp,eq],['原图','灰度 rgb_to_grayscale','高斯模糊 gaussian_blur2d','Sobel 梯度','Canny 边缘','色相旋转 HSV','锐化 sharpness','直方图均衡 equalize'],'images/02-filters.png')
print('canny edges pixels:',int(edges.sum()))
