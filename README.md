# if running this on google
change markdown cells to python

# to adjust configs
go to the cell with header #config

### main configs
SOURCE_DIR: path to directory containing all .dcm and .npy files  
if empty, assumes files already exist in target directory  
*str*

TARGET_DIR: path to directory that will contain all outputs  
default=current directory  
*str*

COPY_DATA_TO_TARGET: if .dcm and .npy files should be copied from source to target  
*True or False*

### subdirectories - model
MODEL_DIR: subdirectory in target that contains saved models  
*str*

MODEL: save model as, always ends with .keras  
*str*

### subdirectories - data
following subdirectories within target directory  
*str*  

TRAIN_IMGS: will contain training .dcm files if COPY_DATA_TO_TARGET=True  
TRAIN_MSKS: will contain training .npy files if COPY_DATA_TO_TARGET=True  
VAL_IMGS: will contain validation .dcm files if COPY_DATA_TO_TARGET=True  
VAL_MSKS: will contain validtion .npy files if COPY_DATA_TO_TARGET=True  
TEST_IMGS: manually copy .dcm files to this directory for testing  
TEST_MSKS: manually copy .npy files to this directory for testing, optional

### data
R_TRAIN: ratio of training images to all images  
*float between 0.0 and 1.0*

R_VAL: ratio of validation images to all images  
*float between 0.0 and 1.0*

### image and loading
IMG_WIDTH: number of pixels in image width  
if an image does not match this dimension, it will be forced into size   
*int*

IMG_HEIGHT: number of pixels in image height  
if an image does not match this dimension, it will be forced into size   
*int*

IMG_CHANNELS: number of channels in image  
assume 1 channel for grayscale xray dicoms  
*int*

BATCH_SIZE: number of images to load at once in pipeline  
assume 8 or 16 for cpu, 16 or 32 for gpu  
*int*

BUFFER_SIZE: number of images to keep loaded at all times  
data in buffer is shuffled, so larger the better, but results in slower training  
*int*

### model
STEP: learning rate for model, adjusted automatically during training by Adam optimization algo  
assume somewhere between 1e-4 and 1e-2  
*float*

EPOCHS: number of passes through training set  
allows model to learn features of training set iteratively, too many epochs results in overfit  
*int*

LAYERS: number of encoder and decoder blocks for unet architecture  
assume 4, too low and model is a failure, too high and model takes exponentially longer to train  
*int*

# to test a trained model
make sure trained .keras model file is in TARGET_DIR/MODEL_DIR with appropriate MODEL name  
adjust TEST_IMAGE and TEST_MASK in final cell with header #make prediction  
TEST_MASK is optional  
run the #make prediction cell  
optionally, save its output  
currently, can only read 1-channel .dcm files

# for segmentation.py
edit image configs and variables as needed in .py file  
run from command line  
>>> python segmentation.py path/to/input/image -model path/to/model -save path/to/output/dcm  
saves overlay of original image and model probability as .dcm file  
