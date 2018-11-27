#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from skimage import io

import matplotlib.animation as ani
from IPython.display import HTML
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 1000
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


source_dir = Path('../data/source/test_img')
target_dir = Path('../results/target/test_latest/images')


# In[ ]:


source_img_paths = sorted(source_dir.iterdir())
target_synth_paths = sorted(target_dir.glob('*synthesized*'))
target_label_paths = sorted(target_dir.glob('*input*'))

len(source_img_paths) == len(target_synth_paths) == len(target_label_paths)


# In[ ]:


def animate(nframe):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    source_img = io.imread(source_img_paths[nframe])
    ax1.imshow(source_img)
    ax1.set_xticks([])
    ax1.set_yticks([])

    target_label = io.imread(target_label_paths[nframe])
    ax2.imshow(target_label)
    ax2.set_xticks([])
    ax2.set_yticks([])

    target_synth = io.imread(target_synth_paths[nframe])
    ax3.imshow(target_synth)
    ax3.set_xticks([])
    ax3.set_yticks([])

#fig = plt.figure(figsize=(12, 6))
fig=plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

anim = ani.FuncAnimation(fig, animate, frames=len(target_label_paths), interval=1000/24)
plt.close()


# In[ ]:


js_anim = HTML(anim.to_jshtml())
js_anim


# In[ ]:


anim.save("output.gif", writer="imagemagick")


# In[ ]:




