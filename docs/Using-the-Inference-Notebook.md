An inference notebook is available in the root directory of the repository. You can run it locally or through Google Colaboratory without any major modifications of the script.

# Index
1. [Requirements](#1-requirements)
   * [Google Colaboratory specific requirements](#google-colaboratory-specific-requirements)
   * [Local usage requirements](#local-usage-requirements)
2. [Configuration of the notebook](2-configuration-of-the-notebook)
   * [Working with Google Colaboratory](#working-with-google-colaboratory)
   * [Working Locally](#working-locally)
   * [Common configuration](#common-configuration)
      * ["Configuration" cell](#configuration-cell)

# 1. Requirements
At that time, you will at least need the following resources to run the inference notebook : 
* One or more weights file(s) (a weights file is a folder containing an `assets`, `variables` folder and a `saved_model.pb` file, they might be zipped so you need to unzip it) in the same folder as the different tools. Images should be put in a sub-directory named `images\<Mode name>`;

* An image to start an inference, it should be placed in a directory named `images` in a subdirectory named as the inference mode you want to use with it (for example, a subdirectory names `main`).

  To get performance metrics for an image, you will also need an annotations file (see [Supported formats](Annotations-format-compatibility.md#supported-formats)) having the same name as the image in the same folder.

## Google Colaboratory specific requirements
You will find specific requirements for Google Colaboratory in the [Using the Inference Notebook with Google Colaboratory](Using-the-Inference-Notebook-with-Google-Colaboratory.md#specific-requirements) page.

## Local usage requirements
You will need the Python environment installed (see [Installation Guide](Installation-Guide.md) [[FR ver.](Guide-d'installation.md)]).

# 2. Configuration of the notebook
## Working with Google Colaboratory
You will find the specific configuration explanation for Google Colaboratory in the [Using the Inference Notebook with Google Colaboratory](Using-the-Inference-Notebook-with-Google-Colaboratory.md#configuration-of-the-notebook) page.


## Working Locally
Simply start Jupyter via the shortcut created during the [Installation Guide](Installation-Guide.md) [[FR ver.](Guide-d'installation.md)] or by starting an Anaconda (or Miniconda) prompt, navigating to the repository folder using `cd` command, activating the environment via `conda activate Skinet`, and finally by starting Jupyter using `jupyter notebook`.


## Common configuration
### 'Configuration' cell
```Python
mode = 'chain'
displayMode = "All steps" #@param ["All steps", "Only statistics"]

forceFullSizeMasks = False  #@param {type:"boolean"}
forceLowMemoryUsage = False  #@param {type:"boolean"}
saveResults = True #@param {type:"boolean"}
saveDebugImages = False #@param {type:"boolean"}

moveAutoCleanedImageToNextMode = True #@param {type:"boolean"}
```

These parameters will control which mode the Inference Tool will run and how much text will be displayed.


* `mode`: The name of the inference mode to use. Note that 'chain' will use multiple modes. 
* `displayMode`: Whether to display every step (`"All steps"`) of an inference, or only statistics (`"Only statistics"`). 



Next parameters are booleans variables so they can take `True` or `False` as value:


* `forceFullSizeMasks`: Whether to force usage of full-sized masks instead of mini-masks.

* `forceLowMemoryUsage`: Force low-memory mode to use the least RAM as possible.

* `saveResults`: Whether to save the results or not.

* `saveDebugImages`: Whether to save debug images between post-processing methods.

    


* `moveAutoCleanedImageToNextMode`: Whether to move automatically cleaned image image into next mode input folder.
