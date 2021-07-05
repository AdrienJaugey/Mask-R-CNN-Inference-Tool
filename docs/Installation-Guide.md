To use the available tools in this repository, you will need to install a few things first. Following is a little guide that explains how to install what is required to run the Inference Notebook, the Training Notebook or the ```datasetFormator.py``` script. This guide is made for Windows, you may have to check online how to install CUDA Toolkit, cuDNN and how to make a shortcut/script to start a notebook for your OS.

# Index
1. [Getting all the required tool's files](#1-getting-all-the-required-tools-files)
2. [Setting up the Python Environment](#2-setting-up-the-python-environment)
3. [Installing CUDA Toolkit and cuDNN](#3-installing-cuda-toolkit-and-cudnn)
4. [Making a shortcut to easily open the tools](#4-making-a-shortcut-to-easily-open-the-tools)

# 1. Getting all the required tool's files
1. [Download](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/archive/master.zip) or clone the [repository](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition.git).
2. Unzip or move the repository in the directory of your choice.
3. Download the weights file (.h5) and maybe some images you want to run the inference on and put it/them in the same folder. Images should be put in a sub-directory named ```images``` (this can be costumed).

# 2. Setting up the Python Environment
1. Download and install [MiniConda3](https://conda.io/en/latest/miniconda), you can also use [Anaconda3](https://www.anaconda.com/products/individual#Downloads).
2. Start **Anaconda Prompt** using **Start Menu** or **Windows Search Bar**.  
3. Using the console, move to the same directory as step 2. 
    * To change directory, use ```cd <Directory Name>``` command.
    * To switch the drive you are using, just write its letter adding ":" and then press ENTER (for example, to switch from C drive to D drive, write ```D:``` and press ENTER).  
4. Execute the following command: ```conda env create -f environment.yml```.

# 3. Installing CUDA Toolkit and cuDNN
Using a CUDA-capable GPU that supports CUDA 9.0 (please refer to [CUDA GPUs list](https://developer.nvidia.com/cuda-gpus) to know if your GPU as a ```Compute Capability``` of at least 3.0, Turing or newer architecture will require that you compile TensorFlow with an higher CUDA version) will considerably accelerate training.

1. Download and install [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive).
2. Download and install [cuDNN 7.0.5 for CUDA 9.0](https://developer.nvidia.com/rdp/cudnn-archive) ([Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)).

# 4. Making a shortcut to easily open the tools
1. Using **Start Menu** or **Windows Search Bar**, right click on **Anaconda Prompt** and click on ```Open file location```.
2. In the file explorer, right click on **Anaconda Prompt** shortcut and then click ```Send To```, and finally click ```Desktop (create shortcut)```.
3. Go to your Desktop. You can move the shortcut wherever you want, it will be used to access the tool, so put it where you can easily access.
4. Right click on the shortcut and then click on ```Properties```.
5. Set the **Start in** path to the directory which contains the downloaded repository from the [first part of this installation guide](#1-getting-all-the-required-tools-files) part.
6. In the **Target** field, replace ```C:\Users\<USER>\miniconda3``` with ```Skinet && jupyter notebook```.
7. (OPTIONAL) To start a specific notebook directly from the shortcut, just add the notebook file name at the end of the **Target** field. This should look like ```activate.bat Skinet && jupyter notebook MyNotebook.ipynb```
8. (OPTIONAL) The shortcut icon can be changed to something that fits better.
9. Click ```OK```

The installation should be done. You may try the Inference Notebook to be sure everything works fine.