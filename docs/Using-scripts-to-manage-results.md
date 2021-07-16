# Using scripts to manage results

## Index

- [Introduction](#introduction)
1. [extract_files_from_results.py script](#1-extract_files_from_resultspy-script)
2. [gather_stats(histo)_into_csv.py scripts](#2-gather_statshisto_into_csvpy-scripts)
3. [remove_non_failed.py script](#3-remove_non_failedpy-script)
4. [TF OD API's custom scripts](#4-tf-od-apis-custom-scripts)
    1. [create_skinet_tf_record.py script](#1-create_skinet_tf_recordpy-script)
    2. [test_record.py script](#2-test_recordpy-script)

## Introduction

​	Different scripts are available in this repository in order to manage a lot of results files, or prepare files needed for training of Mask R-CNN Inception ResNet V2. These scripts are located in the [scripts](../scripts) folder. To use these scripts :

- Using **Start Menu** or **Windows Search Bar**, open **Anaconda Prompt**;
- Using the `cd` command, move to the repository folder;
- Activate the Python environment created in [Installation Guide §2](Installation-Guide.md#2-setting-up-the-python-environment) by using `conda activate Skinet` command;

​	For each script (except for the ones located in [scripts/TF OD API](../scripts/TF OD API)), you should be able to display the help message, describing the purpose of the script and the syntax to use for calling it, using :

```shell
python scripts/<name of the script>.py -h
```

:information_source: If you are on Windows, you will should use back-slashes (`\`) instead of forward-slashes (`/`) in commands, but it should still work if you do not use them.

## 1. extract_files_from_results.py script

​	This script is used to copy (default) or move a type of result file (annotation file, original/predicted (clean or not)/expected image, ...) from all the images that went through the Inference Tool during the same execution.

Usage: 

python scripts/extract_files_from_results.py [-h] [--dst DST] [--extension {jpg, png, jp2}] [--annotation_format {xml, json}]
      																[--mode {copy, move}] [--chain] [--inference_mode MODE]  src
      														        {original_image, cleaned_image, expected, predicted, predicted_clean, stats, histo,
																	  annotation}
positional arguments:
  src                   																	Path to the directory containing results of the inferences
  {original_image, cleaned_image, expected, predicted, 	 Which type of file
  predicted_clean, stats, histo, annotation}

optional arguments:
  -h, --help            													   Show this help message and exit
  --dst DST, -d DST    												  Path to the directory containing results of the inferences
  --extension {jpg, png, jp2}, -e {jpg, png ,jp2}			Format of the images
  --annotation_format {xml, json}, -a {xml, json}	      Format of the annotation
  --mode {copy, move}, -m {copy, move}					Whether the images should be copied or moved
  --chain, -c           													   Whether the results are from chain mode or not
  --inference_mode MODE, -i MODE						   Specific inference mode of chain results. Disable --chain argument 

> Let's say you want to move all JSON annotations files from the `mode1` mode of the `results/inference_2021-16-07_12-00-00` folder, which has been created using chain inference mode, to the `extracted/` folder. The command you would use is :
>
> ```shell
> python scripts/extract_files_from_results.py results/inference_2021-16-07_12-00-00 annotation -m move -i mode1 -a json -d extracted
> ```

:information_source: Cleaned image may not be retrieved for now due to dynamic names being a recent change​

## 2. gather_stats(histo)_into_csv.py scripts

​	Those two scripts allow you to gather a lot of statistics (or histograms) files from the same folder into a CSV file (that can be easily opened by Excel or other spreadsheet software). To gather statistics (or histograms) from results folders, you should extract all the needed files you want to gather into a folder, using the previous script. 

Usage:

python scripts/gather_stats_into_csv.py [-h] src [dst]
python scripts/gather_histo_into_csv.py [-h] src [dst]

positional arguments:
  src         path to the directory containing statistics (or histograms) files
  dst         path to the output csv file

optional arguments:
  -h, --help  show this help message and exit

> Let's say you want to gather histograms of the `histos/` folder into `results_data/histos.csv` file, the command to use would be :
>
> ```shell
> python scripts/gather_histo_into_csv.py histos/ results_data/histos.csv
> ```

## 3. remove_non_failed.py script

​	This script allows you to move non failed images from a folder to another, using the `failed.json` file created if at least an image has failed.

Usage: 

python scripts/remove_non_failed.py [-h] failed_list [src] [dst]

positional arguments:
  failed_list  path to the json file containing path of images that have failed
  src          path to the directory containing original images and annotations
  dst          path to the directory in which moving the images and annotations

optional arguments:
  -h, --help   show this help message and exit

> To move successful images from the input folder `images/mode1` to the `images/mode1/successful` folder, using `results/inference_2021-16-07_12-00-00/failed_list.json` as failed list, the command would be :
>
> ```shell
> python scripts/remove_non_failed.py results/inference_2021-16-07_12-00-00/failed_list.json images/mode1 images/mode1/successful
> ```

## 4. TF OD API's custom scripts

To use the following scripts, you need to use the same Python environment you used when installing the TF OD API.

### 1. create_skinet_tf_record.py script

​	This script allows you to convert the dataset format generated by the Dataset Generator Notebook to [TF Records](https://www.tensorflow.org/tutorials/load_data/tfrecord) files used to train Mask R-CNN Inception ResNet V2 using the TF OD API.

> Let's say you want to convert mode1's train and val datasets using the `mode1_label_map.pbtxt` [label map](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#example-image) file and save the generated files to `records/mrcnn_mode1_train(val).tfrecord`. To generates both `train` and `val` datasets at the same time from `mrcnn_mode1_dataset_train(val)` folders, you can use `%TYPE%` keyword where there should be `train`/`val` in the input/output paths. The command you could use is : 
>
> ```shell
> python "scripts/TF OD API/create_skinet_tf_record.py" --data_dir="mrcnn_mode1_dataset_%TYPE%" --output_path="records/mrcnn_mode1_%TYPE%.tfrecord" --label_map_path="mode1_label_map.pbtxt" --log_path="record_generator.log"
> ```

:information_source: If the script does not work, you may want to try to move it to `models/research/object_detection/dataset_tools` and retry.

### 2. test_record.py script

​	You can test the generated TF Records using the following script. A window will display each image with annotated objects (some might not appear) for you to check that the conversion went good and you can start the training.

Usage: 

python "scripts/TF OD API/test_tfrecord.py" [-h] label_map_proto_file tfrecord_path [sleep_time]

positional arguments:
  label_map_proto_file  path to the label map file
  tfrecord_path         Path to the TFRecord files. You can use .tfrecord-?????-of-00000 format to loop through each shard
  sleep_time             Time before reading the next TFExample in ms

optional arguments:
  -h, --help            show this help message and exit

> If you want to loop through each of the five (or any other number) TF Records shards (files) of the `train` dataset of `mode1` mode, you should use following command :
>
> ```shell
> python "scripts/TF OD API/test_tfrecord.py" mode1_label_map.pbtxt "records/mrcnn_mode1_train.tfrecord-?????-of-00005" 500
> ```

:information_source: Some annotated objects might not appear, it seems to be a visual bug of the TF OD API, objects are still there. 