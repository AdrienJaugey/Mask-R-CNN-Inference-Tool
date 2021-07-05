# Index
1. [Specific requirements](#1-specific-requirements)
2. [Configuration of the notebook](#2-configuration-of-the-notebook)

# 1. Specific requirements
You will need a compatible python environment to use the notebook. Having a [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) is required.

The development and training environment was the following :
* Python 3.5 (using Anaconda Interpreter)
* [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive) (see [installation instructions](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Installation-Guide#3-installing-cuda-toolkit-and-cudnn))
* [cuDNN 7.0](https://developer.nvidia.com/rdp/cudnn-archive) (see [installation instructions](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Installation-Guide#3-installing-cuda-toolkit-and-cudnn))
* Python Librairies ([environment.yml](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/environment.yml) file is available to [create the Anaconda environment](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Installation-Guide#2-setting-up-the-python-environment)):
  * cython==0.29.16
  * h5py==2.10.0
  * html5lib==0.9999999
  * imageio==2.8.0
  * imagesize==1.2.0
  * imgaug==0.4.0
  * ipython==7.9.0
  * jsonschema==3.2.0
  * jupyter==1.0.0
  * keras==2.1.6
  * matplotlib==3.0.3
  * numpy==1.18.3
  * opencv-python==4.2.0.34
  * pillow==7.1.1
  * pip==20.0.2
  * scikit-image==0.15.0
  * scipy==1.1.0
  * tensorboard==1.7.0
  * tensorflow-gpu==1.7.0

[EdjeElectronics' tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment) has been used and combined with Mask R-CNN dependencies to create the Anaconda Environment.

You can find an installation guide [here](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Installation-Guide).

# 2. Configuration of the notebook
There is no real specific configuration for the local usage of the notebook. However, you should create an _images_ directory in the directory where the notebook is located and keep the same structure as this repository (_mrcnn_ and _datasetTools_ directories with their respective files, _nephrology.py_ at the same place than the notebooks...). Then put the image(s) you want to start the inference on into the _images_ directory.
