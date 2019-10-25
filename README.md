
# Characterizing Bias in Classifiers using GenerativeModels



## Code:

The folders GAN_Code and Sampling_Code contain the code used to generate our results.  Each folder contains it's own ReadMe with further details. 

## Data:

We release our selected MSCeleb subset for training, one can download the 
data from the link:
https://drive.google.com/open?id=1xZ622AXiwhr4apzoM_M7M2yKi8MyqEVq
All the images are stored in 'balanced_128_0325.pickle', and the according label
are stored in 'balanced_multi_label_0325.pickle'.
To train the model, the original data need to processed into *.tfrecods format.
run: 'python dataset_tool.py create_from_images datasets/face <path of the data>'


