Using the Notebook Inference on Google Colaboratory will ask you a bit more of configuration. Common requirements and configuration are provided in the [Using the Inference Notebook](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/wiki/Using-the-Inference-Notebook) page. 

# Index
1. [Specific requirements](#1-specific-requirements)
   * [Getting the files easily without upload](#getting-the-files-easily-without-upload)
2. [Configuration of the notebook](#2-configuration-of-the-notebook)
   * ["Connecting to Google Drive" cell](#connecting-to-google-drive-cell)
   * ["Retrieving your image(s)" cell](#retrieving-your-images-cell)
   * ["By copy from Google Drive" and "Retrieving Weights File" cells](#by-copy-from-google-drive-and-retrieving-weights-file-cells)


# 1. Specific requirements
To run the notebook on Google Colaboratory, you will need a google account and uploading the required files somewhere in your Google Drive folder.

## Getting the files easily without upload
You can get the files directly in your Google Drive folder without having to upload it, you will need to know someone that can add you (with a Google Account, not an access link) to a Google Drive folder containing the wanted files and give you edit permissions. 

1. Once you have access to the folder, make a copy of the wanted file (for example, the weights file as it is quite heavy). _Be sure that the person has **paused** its syncing client if it has **Backup and Sync from Google** installed on a running computer_.

2. Right-click on the copy and move it to your personal Google Drive folder. _Once it is done, the other person can reactivate the syncing client if needed._

3. You can rename the file that is in your Google Drive and use it with the notebook from now on.

4. Repeat steps 1 to 3 for each file or folder you have to get.

# 2. Configuration of the notebook
A few variables have to be set in order to be able to run the notebook on Google Colaboratory. These are essentially paths to the needed files in Google Drive.

## "Connecting to Google Drive" cell
You can do this during the first execution, after completing all other configurations.  
The first time this cell runs, a link will be prompted to allow Google Colaboratory to access your Google Drive folders. Follow the link, choose the account containing the required files (at least the weights file) and accept. Then copy the given link to the input text field under the first link you followed.

## "Retrieving your image(s)" cell
In this cell, you just have to choose the way you want to import your image(s) (and annotations files). Use the dropdown list on the right to choose if you want to upload directly the file(s) or if you want to import it/them from a Google Drive folder.

## "By copy from Google Drive" and "Retrieving Weights File" cells
If you chose to get the file(s) from Google Drive, you may have to customize variables of this cell. As explained in the Notebook, you want to customize ```customPathInDrive```, ```imageFilePath``` and ```weightFileName``` variables so they represent the path to your file(s) in your Google Drive.  

Let's say you have this hierarchy in your Google Drive:
```
Root directory of Google Drive
  ├─── Directory1
  └─── Directory2
       ├─── images
       │    ├─── example1.jp2
       │    ├─── example1.xml
       │    ├─── example2.png
       │    └─── example2.json
       └─── saved_weights
            └─── weights.h5
```

1.   ```customPathInDrive``` must represent all the directories between the root directory and your image file. In the example, it would be ```Directory2/images/```. **Do not forget the final `/`** if you have to use this variable;
  
2.   ```imageFilePath``` must represent the path to the file you want to upload. In the example, it would be ```example1.jp2```. It can also be empty if you want to import all the folder's images directly to Google Colab, so in the example ```example1.png``` and ```example2.png``` would be imported. The script look for images with a ```jp2``` or ```png``` extension.

3. If ```annotationsFile``` checkbox is checked the script will try to import the annotation file by looking for a file with the same name as the image and ```xml``` or ```json``` extension. It works whatever you are importing only one image or a complete folder of images.

Use the text fields available on the right of the cell to set these variables.

The "Retrieving Weights File" cell works the same. ```weightFileName``` is the equivalent of ```imageFilePath```, the difference is that this variable is required.