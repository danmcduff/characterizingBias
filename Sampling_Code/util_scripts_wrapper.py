
import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc

import configWrapper as config
import misc
import tfutil
import trainWrapper as train
import dataset

#----------------------------------------------------------------------------
# fuse multi-label (race and gener) to single label, and set threshhold for each category
def parse_label(r, g):
    if g == '0':
        if r == '0':
            label = '0'
            thr = -4600
        if r == '1':
            label = '1'
            thr = -4200
        if r == '2':
            label = '2'
            thr = -4900
        if r == '3':
            label = '3'
            thr = -4800
    if g == '1':
        if r == '0':
            label = '4'
            thr = -3300
        if r == '1':
            label = '5'
            thr = -5000
        if r == '2':
            label = '6'
            thr = -5000
        if r == '3':
            label = '7'
            thr = -5500
    return int(label), thr
    

def generate_fake_images(Gs, D, random_state, race, gender, num_pngs = 1, grid_size=[1,1], image_shrink=1, png_prefix=None, random_seed=1000, minibatch_size=8):
        
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = misc.random_latents(np.prod(grid_size), Gs, random_state=random_state)

        trans_label, thr = parse_label(race, gender)
        labels = np.zeros([latents.shape[0], 8], np.float32)
        labels[:, trans_label] = 1.0 
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)

        score, _ = D.run(images)
        if score >= thr:
            # output image
            img = images[0].transpose(1,2,0)
            scipy.misc.imsave('test%d.jpg' %png_idx, img)
            print('save image %d' %png_idx)
                                    

