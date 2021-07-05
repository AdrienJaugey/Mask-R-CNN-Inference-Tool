An inference notebook is available in the root directory of the repository. You can run it locally or through Google Colaboratory without any major modifications of the script.

# Index
1. [Requirements](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#1-requirements)
   * [Google Colaboratory specific requirements](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#google-colaboratory-specific-requirements)
   * [Local usage requirements](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#local-usage-requirements)
2. [Configuration of the notebook](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#2-configuration-of-the-notebook)
   * [Working with Google Colaboratory](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#working-with-google-colaboratory)
   * [Working Locally](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#working-locally)
   * [Common configuration](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#common-configuration)
      * ["Initialisation" cell](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook#initialisation-cell)

# 1. Requirements
At that time, you will at least need the following resources to run the inference notebook : 
* A weights file (`.h5`);
* An image to start an inference.
To get performance metrics for an image, you will also need an annotations file (see [Supported formats](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Annotations-format-compatibility#supported-formats)) having the same name as the image

## Google Colaboratory specific requirements
You will find specific requirements for Google Colaboratory in the [Using the Inference Notebook with Google Colaboratory](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook-with-Google-Colaboratory#specific-requirements) page.

## Local usage requirements
You will find the specific requirements for local usage in the [Using the Inference Notebook Locally](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook-Locally#specific-requirements) page.

# 2. Configuration of the notebook
## Working with Google Colaboratory
You will find the specific configuration explanation for Google Colaboratory in the [Using the Inference Notebook with Google Colaboratory](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook-with-Google-Colaboratory#configuration-of-the-notebook) page.


## Working Locally
You will find the specific configuration explanation for local usage in the [Using the Inference Notebook Locally](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook-Locally#configuration-of-the-notebook) page.


## Common configuration
### 'Initialisation' cell
In this cell, you will find variables that will customize the model behavior, by defaults they are set to values than provided good results so you can experiment with changing them but you may get worse results. Google Colaboratory users will easier configuration process as you can use input text fields, sliders and checkboxes instead of writing everything in Jupyter.  
* Inference variables : 
  * ```MODEL_PATH```: this variable should be the same as the previous ```weightFileName``` variable if you are using Google Colaboratory, representing the name of the weights file. If you are using the notebook locally, this variable is the path to the weights file which can be in another directory than the notebook itself. If you have imported more than one weights file, you can switch the file that is used here, this has to be done between executions of course;
  * ```RESULTS_PATH```: this variable represents the path to the results directories. The last directory in this path will be created each execution, in order not to overwrite previous results, a timestamp will be added to the name of the folder. For example, the default ```results/inference/``` path will give the ```results/inference_20200527_143042``` directory if the inference started on the 27th of May 2020 at 2:30:42 pm.
  * ```DIVISION_SIZE```: this variable represents the side length of the square divisions that are used to divide the image. The model uses divisions to keep as many details on the images as it can.
  * ```saveResults```: this variable determines whether the model should save the results or not. It may be useful if you only want the predicted annotations and/or the average precision of the model for an image, without getting the predicted image and confusion matrix.
* Post-processing variables:
  * ```FUSION_BB_THRESHOLD```: this threshold controls the first selection step of the results fusion process. It represents the least part of one of two bounding boxes being included in the other one to pass the step. 
  * ```FUSION_MASK_THRESHOLD```: this threshold controls the last selection step of the results fusion process. It represents the least part of one of two masks being included in the other one for them to be fused.
  * ```FILTER_BB_THRESHOLD```: this threshold controls the second selection step of the results filtering process, the first step being that the classes of the predicted elements has a different priority level. It represents the least part of one of two bounding boxes being included in the other one to pass the step.
  * ```FILTER_MASK_THRESHOLD```: this threshold controls the last selection step of the results filtering process. It represents the least part of one of two masks being included in the other one for them to be filtered.