
# coding: utf-8

# ##### Download and extract video

# In[1]:


import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#from pytube import YouTube
from pathlib import Path


# In[2]:


save_dir = Path('../data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)



# In[3]:


# Bruno Mars - That's What I Like
#yt = YouTube('https://www.youtube.com/watch?v=PMivT7MJ41M')
#yt.streams.first().download(save_dir, 'mv')


# In[4]:


# cap = cv2.VideoCapture('afro.mp4')
# i = 0
# z = 1
# j= 630
# while(cap.isOpened()):
#     flag, frame = cap.read()
#     z+=1
#     if z<j:
#         continue
#     if flag == False or i == 5000:
#         break
#     cv2.imwrite(str(img_dir.joinpath(f'img_{i:04d}.png')), frame)
#     print(i)
#     i += 1
# #cv2.destroyAllWindows()
# cap.release()


# # Pose estimation (OpenPose)

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


openpose_dir = Path('../src/pytorch_Realtime_Multi-Person_Pose_Estimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('../src/utils')

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[7]:


# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose


# In[8]:


weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()
#pass


# ## check

# In[9]:


img_path = sorted(img_dir.iterdir())[1]
#print(img_path)
img = cv2.imread(str(img_path))
shape_dst = np.min(img.shape[:2])
# offset
oh = (img.shape[0] - shape_dst) // 2

#Yijun modified width
ow = (img.shape[1] - shape_dst) // 2

img = img[oh:oh+shape_dst, ow:ow+shape_dst]
img = cv2.resize(img, (512, 512))

#plt.imshow(img[:,:,[2, 1, 0]]) # BGR -> RGB


# In[23]:


multiplier = get_multiplier(img)
with torch.no_grad():
    paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')

r_heatmap = np.array([remove_noise(ht)
                      for ht in heatmap.transpose(2, 0, 1)[:-1]])\
                     .transpose(1, 2, 0)
heatmap[:, :, :-1] = r_heatmap
param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
label = get_pose(img, param, heatmap, paf)

#plt.imshow(label)


# In[ ]:


# ## make label images for pix2pix

# In[11]:


train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)
#train_face_dir=face_dir.joinpath('train_img')
#train_face_label_dir=face_dir.joinpath('train_label')

for idx in tqdm(range(0,500)):
    img_path = img_dir.joinpath(f'img_{idx:04d}.png')
    img = cv2.imread(str(img_path))
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh+shape_dst, ow:ow+shape_dst]
    img = cv2.resize(img, (512, 512))
    multiplier = get_multiplier(img)
    with torch.no_grad():
        paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
    r_heatmap = np.array([remove_noise(ht)
                      for ht in heatmap.transpose(2, 0, 1)[:-1]])\
                     .transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    #label = get_pose(img,param, heatmap, paf)
    #face,facial,pos=get(img)
    #cv2.inwrite(str(train_face_dir.joinpath(f'img_{idx:04d}.png')),face)
    #cv2.imwrite(str(train_face_label_dir.joinpath(f'label_{idx:04d}.png')), label)
    cv2.imwrite(str(train_img_dir.joinpath(f'img_{idx:04d}.png')), img)
    cv2.imwrite(str(train_label_dir.joinpath(f'label_{idx:04d}.png')), label)

torch.cuda.empty_cache()
