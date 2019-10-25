
# Characterizing Bias in Classifiers using Generative Models



## Code:

The folders GAN_Code and Sampling_Code contain the code used to generate our results.  Each folder contains it's own ReadMe with further details. 

### GAN Code:
(This work is modified upon open source code from NVIDIA CORPORATION - https://github.com/tkarras/progressive_growing_of_gans)

Install all dependency libraries:
'pip install requirements.txt'

Synthesize images:

'python trainWrapper.py'

One can specifiy any 'race', 'gendere' and 'images numbers' desired
to function 'util.generate_fake_images()'

Train model from scratch:

Prepare data:

We release our selected MSCeleb subset for training, one can download the 
data from the link:
https://drive.google.com/open?id=1xZ622AXiwhr4apzoM_M7M2yKi8MyqEVq

All the images are stored in 'balanced_128_0325.pickle', and the according label
are stored in 'balanced_multi_label_0325.pickle'.

To train the model, the original data need to processed into *.tfrecods format.

run: 'python dataset_tool.py create_from_images datasets/face <path of the data>'

Train network

run: 'python train.py'

### Sampling Code:

This is the code for running the sampling algorithm to interogate a classifier or API:

In order to run the code you will need API keys for the corresponding API or a classifier model you want to interrogate.  The specific model or API you chose will impact the dependencies of this code.  

cd [path to folder]/sampling_Code
python NeurIPS_Sampling_Code.py



## Data:

We release our selected MSCeleb subset for training, one can download the 
data from the link:
https://drive.google.com/open?id=1xZ622AXiwhr4apzoM_M7M2yKi8MyqEVq
All the images are stored in 'balanced_128_0325.pickle', and the according label
are stored in 'balanced_multi_label_0325.pickle'.
To train the model, the original data need to processed into *.tfrecods format.
run: 'python dataset_tool.py create_from_images datasets/face <path of the data>'


