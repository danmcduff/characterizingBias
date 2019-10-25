
# This work modified upon open source code from NVIDIA CORPORATION.  
# https://github.com/tkarras/progressive_growing_of_gans

import os
import time
import numpy as np
import tensorflow as tf

import configWrapper as config
import tfutil
import dataset
import misc
import util_scripts_wrapper as util

    
    
#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    #-----------------
    network_pkl = misc.locate_network_pkl(14, None)
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(14, None)    
    random_state = np.random.RandomState()

    print('Synthesizing images... ') # synthesize images and specify race and gender
    # util.generate_fake_iamges(Gs, D, random_state, <race>, <gender>, <number of images one want to generate>)
    util.generate_fake_images(Gs,D, random_state, '0', '0', 100)    
#----------------------------------------------------------------------------
