# [DEPRECATED]

A training notebook is available in the root directory of the repository. You can run it locally.

# Index

1. [Requirements](#1-requirements)
2. [Configuration of the notebook](#2-Configuration-of-the-Training-Tool)

# 1. Requirements

At that time, you will at least need the following resources to run the training notebook : 

* A base weights file (`.h5`) to use for transfer learning such as [Matterport's one](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) (automatically downloaded by the Training Notebook);
* A training dataset and validation one 
    * A folder containing a folder per image (same name as image) with subfolders for each classes (same name as the corresponding class) containing masks as binary images (one per mask, 0 for background and 1 [or 255] for mask), and an `images` folder with the input image.
    * Images and annotations from supported formats (ASAP and LabelMe are the only one already implemented) can be used to generate
* Python environment and CUDA (if compatible) installed (Installation Guide ([EN](Installation-Guide.md)|[FR](Guide-d'installation.md)))


# 2. Configuration of the Training Tool

Start the Training Tool by activating the python environment and using the `jupyter notebook Mask_R_CNN_Training_Tool.ipynb` command in the same terminal.



The main parameters are present in the first cell of the notebook. Here, you can customize the mode's name as well as training and validation datasets paths and classes names. You can change other settings such as the weights file used for transfer learning/continuing a training, the size or format of the input images.

```python
# Definition of mode name and training/validation datasets path
mode = "main" 
TRAIN_PATH = 'dataset_train/'
MAP_PATH = 'dataset_val/'

# Part of validation dataset used to compute map
evaluation_size = 1.0 #@param {type:"slider", min:0.01, max:1.0, step:0.01}

# Up to which epoch it should train
NB_EPOCHS = 100

# To restart or continue training, set to last
# If custom is choosed, please set custom_weights_file to true name of the file (should be in ./logs/ directory)
init_with = "coco" #@param ["coco", "imagenet", "last", "base", "custom"]
custom_weights_file = "mask_rcnn_XXXX_XXX.h5"

# Format of images in the dataset
IMAGE_FORMAT = 'jpg' #@param ['jp2', 'png', 'jpg']
# Side size of a division
DIVISION_SIZE = 1024

# Classes definition
if mode == "main":
    CUSTOM_CLASS_NAMES = ["class1", "class2", "class3"]
else:
    raise NotImplementedError(f'Please list classes of {mode} mode.')
```

You can add different modes if you need to alternate between some of them. Add an `elif` statement at the end of the cell with a condition on the name of this new mode. Inside the new case, affect to `CUSTOM_CLASS_NAMES` a list containing the classes name.

When you are done customizing the parameters, you can simply start the notebook by clicking `Cell` then `Run All`.

You will find the output weights files if the `log` directory, under a subfolder with the name of your mode and the date as a sequence of numbers. 
