import json
import os
import re
import traceback
import shutil
import warnings
import gc
from enum import Enum

from datasetTools.AnnotationAdapter import export_annotations
from datasetTools.CustomDataset import SkinetCustomDataset
from mrcnn.Config import Config, DynamicMethod

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from abc import ABC, abstractmethod
    from common_utils import progressBar, formatTime, formatDate, progressText
    from datasetTools.datasetDivider import CV2_IMWRITE_PARAM
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imread, imsave
    from datasetTools import datasetDivider as dD, datasetWrapper as dW, AnnotationAdapter, datasetIsolator as dI

    from mrcnn import utils
    from mrcnn.TensorflowDetector import TensorflowDetector
    from mrcnn import visualize
    from mrcnn import post_processing as pp
    from mrcnn import statistics as stats


def get_ax(rows=1, cols=1, size=8):
    return plt.subplots(rows, cols, figsize=(size * cols, size * rows), frameon=False)


def find_latest_weight(weight_path):
    """
    Return the weight file path with the highest id
    :param weight_path: weight path with %LAST% as the id part of the path
    :return: the weight path if found else None
    """
    if "%LAST%" not in weight_path:
        return weight_path
    folder = os.path.dirname(os.path.abspath(weight_path))
    folder = '.' if folder == "" else folder
    name = os.path.basename(weight_path)
    name_part1, name_part2 = name.split("%LAST%")
    regex = f"^{name_part1}([0-9]+){name_part2}$"
    maxID = -1
    maxTxtID = ""
    for weight_file in os.listdir(folder):
        regex_res = re.search(regex, weight_file)
        if regex_res:
            nbTxt = regex_res.group(1)
            nb = int(nbTxt)
            if nb > maxID:
                maxID = nb
                maxTxtID = nbTxt
    return None if maxID == -1 else weight_path.replace("%LAST%", maxTxtID)


def listAvailableImage(dirPath: str):
    files = os.listdir(dirPath)
    image = []
    for file in files:
        if file in image:
            continue
        name = file.split('.')[0]
        extension = file.split('.')[-1]
        if extension == 'jp2':
            if (name + '.png') not in image and (name + '.jpg') not in image:
                if os.path.exists(os.path.join(dirPath, name + '.jpg')):
                    image.append(name + '.jpg')
                elif os.path.exists(os.path.join(dirPath, name + '.png')):
                    image.append(name + '.png')
                else:
                    image.append(file)
        elif extension in ['png', 'jpg']:
            image.append(file)
        elif extension == 'skinet':
            with open(os.path.join(dirPath, file), 'r') as skinetFile:
                fusionInfo = json.load(skinetFile)
                fusionDir = fusionInfo['image'] + "_fusion"
                if fusionDir in files:
                    divExists = False
                    divPath = os.path.join(fusionDir, fusionInfo['image'] + '_{}.jpg')
                    for divID in range(len(fusionInfo["divisions"])):
                        if fusionInfo["divisions"][str(divID)]["used"] and os.path.exists(
                                os.path.join(dirPath, divPath.format(divID))):
                            image.append(divPath.format(divID))
                            divExists = True
                        elif fusionInfo["divisions"][str(divID)]["used"]:
                            print(f"Div n°{divID} of {fusionInfo['image']} missing")
                    if divExists:
                        image.append(file)

    for i in range(len(image)):
        image[i] = os.path.join(dirPath, image[i])
    return image


class NephrologyInferenceModel:

    def __init__(self, configPath: str, low_memory=False):
        self.__STEP = "init"
        self.__CONFIG = Config(configPath)
        self.__CONFIG_PATH = configPath
        self.__LOW_MEMORY = low_memory
        self.__READY = False
        self.__MODE = None
        self.__MODEL_PATH = None
        self.__MODEL = None

        self.__CLASS_2_ID = None
        self.__ID_2_CLASS = None
        self.__CLASSES_INFO = None
        self.__NB_CLASS = 0
        self.__CUSTOM_CLASS_NAMES = None
        self.__VISUALIZE_NAMES = None
        self.__COLORS = None

        self.__DIVISION_SIZE = None
        self.__MIN_OVERLAP_PART = None
        self.__RESIZE = None

        self.__CONFUSION_MATRIX = None
        self.__APs = None

        self.__PREVIOUS_RES = None

    def load(self, mode: str, forceFullSizeMasks=False, forceModelPath=None):
        # If mode is already loaded, nothing to do
        if self.__MODE == mode:
            print(f"{mode} mode is already loaded.\n")
            return

        self.__MODEL_PATH = find_latest_weight(self.__CONFIG.get_param(mode)['weight_file']
                                               if forceModelPath is None else forceModelPath)

        # Testing only for one of the format, as Error would have already been raised if modelPath was not correct
        isExportedModelDir = os.path.exists(os.path.join(self.__MODEL_PATH, 'saved_model'))
        if isExportedModelDir:
            self.__MODEL_PATH = os.path.join(self.__MODEL_PATH, 'saved_model')

        self.__CLASSES_INFO = self.__CONFIG.get_classes_info(mode)
        self.__CLASS_2_ID = self.__CONFIG.get_mode_config(mode)['class_to_id']

        label_map = {c["id"]: c for c in self.__CONFIG.get_classes_info(mode)}
        if self.__MODEL is None:
            self.__MODEL = TensorflowDetector(self.__MODEL_PATH, label_map)
        else:
            self.__MODEL.load(self.__MODEL_PATH, label_map)
        self.__READY = self.__MODEL.isLoaded()
        if not self.__MODEL.isLoaded():
            raise ValueError("Please provide correct path to model.")

        self.__CONFIG.set_current_mode(mode, forceFullSizeMasks)
        self.__MODE = mode

        self.__DIVISION_SIZE = self.__CONFIG.get_param()['roi_size']
        self.__MIN_OVERLAP_PART = self.__CONFIG.get_param()['min_overlap_part']
        if self.__CONFIG.get_param().get('resize', None) is None:
            self.__RESIZE = None
        else:
            self.__RESIZE = tuple(self.__CONFIG.get_param()['resize'])

        self.__NB_CLASS = len(self.__CLASSES_INFO)
        self.__CUSTOM_CLASS_NAMES = [classInfo["name"] for classInfo in self.__CLASSES_INFO]
        self.__VISUALIZE_NAMES = ['Background']
        self.__VISUALIZE_NAMES.extend([classInfo.get('display_name', classInfo['name'])
                                       for classInfo in self.__CLASSES_INFO])

        self.__COLORS = [classInfo["color"] for classInfo in self.__CLASSES_INFO]

        # Configurations
        self.__CONFUSION_MATRIX = {'pixel': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64),
                                   'mask': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64)}
        self.__APs = []

        mode_config = self.__CONFIG.get_mode_config()
        print()

    def __prepare_image__(self, imagePath, results_path, chainMode=False, silent=False):
        """
        Creating png version if not existing, dataset masks if annotation found and get some information
        :param imagePath: path to the image to use
        :param results_path: path to the results dir to create the image folder and paste it in
        :param silent: No display
        :return: image, imageInfo = {"PATH": str, "DIR_PATH": str, "FILE_NAME": str, "NAME": str, "HEIGHT": int,
        "WIDTH": int, "NB_DIV": int, "X_STARTS": list, "Y_STARTS": list, "HAS_ANNOTATION": bool}
        """
        image = None
        fullImage = None
        imageInfo = None
        image_results_path = None
        roi_mode = self.__CONFIG.get_param()["roi_mode"]
        suffix = ""
        if chainMode and self.__CONFIG.get_previous_mode() is not None:
            image_name, extension = os.path.splitext(os.path.basename(imagePath))
            extension = extension.replace('.', '')
            if extension not in ['png', 'jpg']:
                extension = 'jpg'
            if self.__CONFIG.get_param('previous').get('resize', None) is not None:
                suffix = "_base"
            imagePath = os.path.join(results_path, image_name, self.__CONFIG.get_previous_mode(),
                                     f"{image_name}{suffix}.{extension}")
        if os.path.exists(imagePath):
            imageInfo = {
                'PATH': imagePath,
                'DIR_PATH': os.path.dirname(imagePath),
                'FILE_NAME': os.path.basename(imagePath),
                'HAS_ANNOTATION': False
            }
            imageInfo['NAME'] = imageInfo['FILE_NAME'].split('.')[0]
            if suffix != "":
                imageInfo['NAME'] = imageInfo['NAME'].replace(suffix, "")
            imageInfo['IMAGE_FORMAT'] = imageInfo['FILE_NAME'].split('.')[-1]

            # Reading input image in RGB color order
            imageChanged = False
            if self.__RESIZE is not None:  # If in cortex mode, resize image to lower resolution
                imageInfo['ORIGINAL_IMAGE'] = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = imageInfo['ORIGINAL_IMAGE'].shape
                fullImage = cv2.resize(imageInfo['ORIGINAL_IMAGE'], self.__RESIZE)
                imageChanged = True
            else:
                fullImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = fullImage.shape
            imageInfo['HEIGHT'] = int(height)
            imageInfo['WIDTH'] = int(width)

            if self.__CONFIG.get_param().get('base_class', None) is not None:
                imageInfo['BASE_CLASS'] = self.__CONFIG.get_param()['base_class']

            if 'BASE_CLASS' in imageInfo:
                imageInfo['BASE_AREA'] = height * width
                imageInfo['BASE_COUNT'] = 1

            # Conversion of the image if format is not png or jpg
            if imageInfo['IMAGE_FORMAT'] not in ['png', 'jpg']:
                imageInfo['IMAGE_FORMAT'] = 'jpg'
                imageChanged = True
                tempPath = os.path.join(imageInfo['PATH'], f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                imsave(tempPath, fullImage)
                imageInfo['PATH'] = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                image_results_path = os.path.join(os.path.normpath(results_path), imageInfo['NAME'])
                if chainMode:  # Saving into the specific inference mode
                    image_results_path = os.path.join(image_results_path, self.__MODE)
                if not os.path.exists(image_results_path):
                    os.makedirs(image_results_path, exist_ok=True)
                imageInfo['PATH'] = os.path.join(image_results_path, f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                if not imageChanged:
                    shutil.copy2(imagePath, imageInfo['PATH'])
                else:
                    imsave(imageInfo['PATH'], fullImage)
                if self.__RESIZE is not None:
                    originalImagePath = os.path.join(image_results_path,
                                                     f"{imageInfo['NAME']}_base.{imageInfo['IMAGE_FORMAT']}")
                    if imageInfo['IMAGE_FORMAT'] in imagePath:
                        shutil.copy2(imagePath, originalImagePath)
                    else:
                        imsave(originalImagePath, imageInfo['ORIGINAL_IMAGE'])
            else:
                image_results_path = None

            # If annotations found, create masks and clean image if possible
            annotationExists = False
            if not chainMode:
                for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                    annotationExists = annotationExists or os.path.exists(os.path.join(imageInfo['DIR_PATH'],
                                                                                       imageInfo['NAME'] + '.' + ext))
            if self.__CONFIG.get_param()['allow_empty_annotations']:
                imageInfo['HAS_ANNOTATION'] = annotationExists

            if chainMode and 'BASE_CLASS' in imageInfo and self.__CONFIG.get_previous_mode() is not None:
                baseClassId = self.__CONFIG.get_class_id(imageInfo['BASE_CLASS'], 'previous')
                if self.__PREVIOUS_RES is not None and baseClassId in self.__PREVIOUS_RES['class_ids']:
                    indices = np.arange(len(self.__PREVIOUS_RES['class_ids']))
                    indices = indices[np.isin(self.__PREVIOUS_RES['class_ids'], [baseClassId])]
                    if len(indices) > 0:
                        temp = self.__CONFIG.get_mini_mask_shape('previous')
                        previousModeUsedMiniMask = temp is not None and self.__PREVIOUS_RES['masks'].shape[:2] == temp
                        previousResize = self.__CONFIG.get_param('previous').get('resize', None)
                        fusedMask = np.zeros((height, width), dtype=np.uint8)
                        for idx in indices:
                            mask = self.__PREVIOUS_RES['masks'][:, :, idx].astype(np.uint8) * 255
                            if previousModeUsedMiniMask:
                                bbox = self.__PREVIOUS_RES['rois'][idx]
                                if previousResize is not None:
                                    mask = utils.expand_mask(bbox, mask, tuple(previousResize)).astype(np.uint8) * 255
                                    mask = cv2.resize(mask, (width, height))
                                else:
                                    mask = utils.expand_mask(bbox, mask, (height, width)).astype(np.uint8) * 255
                            elif previousResize:
                                mask = cv2.resize(mask, (width, height))
                            fusedMask = np.bitwise_or(fusedMask, mask)
                        image = cv2.bitwise_and(fullImage, np.repeat(fusedMask[..., np.newaxis], 3, axis=2))

                        crop_to_base_class = self.__CONFIG.get_param().get('crop_to_base_class', False)
                        if crop_to_base_class:
                            fusedBbox = utils.extract_bboxes(fusedMask)
                            image = image[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                            fullImage = fullImage[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                            imsave(imageInfo['PATH'], fullImage)
                            height, width = image.shape[:2]
                            offset = np.array([fusedBbox[0], fusedBbox[1]] * 2)

                        # If RoI mode is 'centered', inference will be done on base-class masks
                        if roi_mode == 'centered':
                            if self.__CONFIG.get_param()['fuse_base_class']:
                                if crop_to_base_class:
                                    imageInfo['ROI_COORDINATES'] = fusedBbox - offset
                                else:
                                    imageInfo['ROI_COORDINATES'] = utils.extract_bboxes(fusedMask)
                            else:
                                imageInfo['ROI_COORDINATES'] = self.__PREVIOUS_RES['rois'][indices]
                                if crop_to_base_class:
                                    imageInfo['ROI_COORDINATES'] -= offset
                            for idx, bbox in enumerate(imageInfo['ROI_COORDINATES']):
                                imageInfo['ROI_COORDINATES'][idx] = dI.center_mask(bbox, (height, width),
                                                                                   self.__DIVISION_SIZE, verbose=0)
                            imageInfo['NB_DIV'] = len(imageInfo['ROI_COORDINATES'])

                        # Getting count and area of base-class masks
                        if self.__CONFIG.get_param()['fuse_base_class']:
                            imageInfo.update({'BASE_AREA': dD.getBWCount(fusedMask)[1], 'BASE_COUNT': 1})
                        else:
                            for idx in indices:
                                mask = self.__PREVIOUS_RES['masks'][..., idx]
                                if previousModeUsedMiniMask:
                                    bbox = self.__PREVIOUS_RES['rois'][idx]
                                    if previousResize is not None:
                                        mask = utils.expand_mask(bbox, mask, tuple(previousResize))
                                        mask = cv2.resize(mask, (imageInfo['HEIGHT'], imageInfo['WIDTH']))
                                    else:
                                        mask = utils.expand_mask(bbox, mask, (imageInfo['HEIGHT'], imageInfo['WIDTH']))
                                imageInfo['BASE_AREA'] += dD.getBWCount(mask)[1]
                                if not self.__CONFIG.get_param()['fuse_base_class']:
                                    imageInfo['BASE_COUNT'] += 1
                                del mask
                        imageInfo['HEIGHT'] = int(height)
                        imageInfo['WIDTH'] = int(width)
            elif annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible",
                          flush=True)
                dW.createMasksOfImage(imageInfo['DIR_PATH'], imageInfo['NAME'], 'data',
                                      classesInfo=self.__CLASSES_INFO, imageFormat=imageInfo['IMAGE_FORMAT'],
                                      resize=self.__RESIZE, config=self.__CONFIG)
                maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                if 'BASE_CLASS' in imageInfo and imageInfo['BASE_CLASS'] in maskDirs:
                    # Fusing masks of base class if needed, then cleaning image using it/them
                    if self.__CONFIG.get_param()['fuse_base_class']:
                        dW.fuseClassMasks('data', imageInfo['NAME'], imageInfo['BASE_CLASS'],
                                          imageFormat=imageInfo['IMAGE_FORMAT'], deleteBaseMasks=True, silent=True)
                    dW.cleanImage('data', imageInfo['NAME'], cleaningClass=imageInfo['BASE_CLASS'],
                                  cleanMasks=False, imageFormat=imageInfo['IMAGE_FORMAT'])
                    maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                    # If RoI mode is 'centered', inference will be done on base-class masks
                    if roi_mode == 'centered':
                        imageInfo['ROI_COORDINATES'] = dI.getCenteredClassBboxes(
                            datasetPath='data', imageName=imageInfo['NAME'], classToCenter=imageInfo['BASE_CLASS'],
                            image_size=self.__DIVISION_SIZE, imageFormat=imageInfo['IMAGE_FORMAT'],
                            allow_oversized=True, config=self.__CONFIG
                        )
                        imageInfo['NB_DIV'] = len(imageInfo['ROI_COORDINATES'])

                    # Getting count and area of base-class masks
                    if imageInfo['BASE_CLASS'] not in self.__CUSTOM_CLASS_NAMES:
                        maskDirs.remove(imageInfo['BASE_CLASS'])
                    imageInfo.update({'BASE_AREA': 0, 'BASE_COUNT': 0})
                    baseClassDirPath = os.path.join('data', imageInfo['NAME'], imageInfo['BASE_CLASS'])
                    for baseClassMask in os.listdir(baseClassDirPath):
                        baseMask = dW.loadSameResImage(os.path.join(baseClassDirPath, baseClassMask),
                                                       fullImage.shape)
                        imageInfo['BASE_AREA'] += dD.getBWCount(baseMask)[1]
                        imageInfo['BASE_COUNT'] += 1
                        del baseMask

                    # If full_images directory exists: image has been cleaned so we have to get it another time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', imageInfo['NAME'], 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        imageInfo['cleaned_image_path'] = os.path.join(imagesDirPath, imageFilePath)
                        image = cv2.cvtColor(cv2.imread(imageInfo['cleaned_image_path']), cv2.COLOR_BGR2RGB)

                for notAClass in ['images', 'full_images']:
                    if notAClass in maskDirs:
                        maskDirs.remove(notAClass)
                # We want to know if image has annotation, if we don't want to detect cortex and this mask exist
                # as we need it to clean the image, we remove it from the mask list before checking if a class
                # we want to predict has an annotated mask
                if not imageInfo['HAS_ANNOTATION']:
                    imageInfo['HAS_ANNOTATION'] = any([d in self.__CUSTOM_CLASS_NAMES for d in maskDirs])

                if imageInfo['HAS_ANNOTATION'] and not silent:
                    print("    - AP and confusion matrix will be computed")

            if roi_mode == "divided" or (roi_mode == 'centered' and 'ROI_COORDINATES' not in imageInfo):
                imageInfo['X_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=width if self.__RESIZE is None else self.__RESIZE[0],
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=0.33 if self.__MIN_OVERLAP_PART is None else self.__MIN_OVERLAP_PART
                )
                imageInfo['Y_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=height if self.__RESIZE is None else self.__RESIZE[0],
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=0.33 if self.__MIN_OVERLAP_PART is None else self.__MIN_OVERLAP_PART
                )
                imageInfo['NB_DIV'] = dD.getDivisionsCount(imageInfo['X_STARTS'], imageInfo['Y_STARTS'])
            elif roi_mode is None:
                imageInfo['X_STARTS'] = imageInfo['Y_STARTS'] = [0]
                imageInfo['NB_DIV'] = 1
            elif roi_mode != "centered":
                raise NotImplementedError(f'\'{roi_mode}\' RoI mode is not implemented.')

        return image, fullImage, imageInfo, image_results_path

    @staticmethod
    def __init_results_dir__(results_path):
        if results_path is None or results_path in ['', '.', './', "/"]:
            lastDir = "results"
            remainingPath = ""
        else:
            results_path = os.path.normpath(results_path)
            lastDir = os.path.basename(results_path)
            remainingPath = os.path.dirname(results_path)
        results_path = os.path.normpath(os.path.join(remainingPath, f"{lastDir}_{formatDate()}"))
        os.makedirs(results_path)
        print(f"Results will be saved to {results_path}")
        logsPath = os.path.join(results_path, 'inference_data.csv')
        with open(logsPath, 'w') as results_log:
            results_log.write(f"Image; Duration (s); Precision; Inference Mode\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, chainMode=False, save_results=True, displayOnlyAP=False,
                  saveDebugImages=False, chainModeForceFullSize=False, verbose=0):

        if len(images) == 0:
            print("Images list is empty, no inference to perform.")
            return

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            results_path, logsPath = self.__init_results_dir__(results_path)
        else:
            print("No result will be saved")
            results_path = None

        if not chainMode:
            self.__CONFUSION_MATRIX['mask'].fill(0)
            self.__CONFUSION_MATRIX['pixel'].fill(0)
            self.__APs.clear()
        total_start_time = time()
        failedImages = []
        for img_idx, IMAGE_PATH in enumerate(images):
            try:
                print(f"Using {IMAGE_PATH} image file {progressText(img_idx + 1, len(images))}")
                image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
                if chainMode:
                    nextMode = self.__CONFIG.get_first_mode()
                    while nextMode is not None:
                        print(f"[{nextMode} mode]")
                        self.load(nextMode, forceFullSizeMasks=chainModeForceFullSize)
                        nextMode = self.__CONFIG.get_next_mode()
                        self.__process_image__(IMAGE_PATH, results_path, chainMode, saveDebugImages, save_results,
                                               displayOnlyAP, logsPath, verbose=verbose)
                else:
                    self.__process_image__(IMAGE_PATH, results_path, chainMode, saveDebugImages, save_results,
                                           displayOnlyAP, logsPath, verbose=verbose)
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                failedImages.append(os.path.basename(IMAGE_PATH))
                print(f"\n/!\\ Failed {IMAGE_PATH} at \"{self.__STEP}\"\n")
                if save_results and self.__STEP not in ["image preparation", "finalizing"]:
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{image_name}; -1; -1;FAILED ({self.__STEP});\n")
        # Saving failed images list if not empty
        if len(failedImages) > 0:
            try:
                with open(os.path.join(results_path, "failed.json"), 'w') as failedJsonFile:
                    json.dump(failedImages, failedJsonFile, indent="\t")
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                print("Failed to save failed image(s) list. Following is the list itself :")
                print(failedImages)

        if len(self.__APs) > 1:
            mAP = np.mean(self.__APs)
            print(f"Mean Average Precision is about {mAP:06.2%}")
            cmap = plt.cm.get_cmap('hot')
            for mat_type in ['mask', 'pixel']:
                for normalized in [False, True]:
                    name = f"Final Confusion Matrix ({mat_type.capitalize()}){' (Normalized)' if normalized else ''}"
                    confusionMatrixFileName = os.path.join(results_path, name.replace('(', '')
                                                           .replace(')', '')
                                                           .replace(' ', '_'))
                    visualize.display_confusion_matrix(self.__CONFUSION_MATRIX[mat_type],
                                                       self.__CUSTOM_CLASS_NAMES.copy(), title=name, cmap=cmap,
                                                       show=False, normalize=normalized,
                                                       fileName=confusionMatrixFileName)
            plt.close('all')
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = f"{mAP:4.3f}".replace(".", ",")
                results_log.write(f"GLOBAL; {total_time}; {mapText}%;\n")

    def __process_image__(self, image_path, results_path, chainMode=False, saveDebugImages=False, save_results=True,
                          displayOnlyAP=False, logsPath=None, verbose=0):
        cortex_mode = self.__MODE == "cortex"
        allowSparse = self.__CONFIG.get_param().get('allow_sparse', True)
        start_time = time()
        self.__STEP = "image preparation"
        image, fullImage, imageInfo, image_results_path = self.__prepare_image__(image_path, results_path, chainMode,
                                                                                 silent=displayOnlyAP)

        if imageInfo["HAS_ANNOTATION"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ground_truth = self.__get_ground_truth__(image_results_path, fullImage,
                                                         imageInfo, displayOnlyAP, save_results)
            if save_results:
                if self.__LOW_MEMORY:
                    del fullImage
                    gc.collect()
                    fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

        debugIterator = -1
        res = self.__inference__(image, fullImage, imageInfo, allowSparse, displayOnlyAP)

        """####################
        ### Post-Processing ###
        ####################"""

        def gather_dynamic_args(method: DynamicMethod):
            dynamic_args = {}
            for dynamic_arg in method.dynargs():
                if dynamic_arg == 'image':
                    dynamic_args[dynamic_arg] = image
                elif dynamic_arg == 'image_info':
                    dynamic_args[dynamic_arg] = imageInfo
                elif dynamic_arg == 'save':
                    dynamic_args[dynamic_arg] = image_results_path
                elif dynamic_arg == 'base_res' and chainMode:
                    dynamic_args[dynamic_arg] = self.__PREVIOUS_RES
                elif dynamic_arg == 'base_res':
                    dynamic_args[dynamic_arg] = self.__get_ground_truth__(image_results_path, fullImage, imageInfo,
                                                                          displayOnlyAP, False, base_gt=True)
                else:
                    raise NotImplementedError(f'Dynamic argument \'{dynamic_arg}\' is not implemented.')
            return dynamic_args

        if len(res['class_ids']) > 0:
            for methodInfo in self.__CONFIG.get_post_processing_method():
                self.__STEP = f"post-processing ({methodInfo['method']})"
                # if saveDebugImages and idx != 0:
                if saveDebugImages:
                    debugIterator += 1
                    self.__save_debug_image__(f"pre_{methodInfo['method']}", debugIterator, fullImage,
                                              imageInfo, res, image_results_path, silent=displayOnlyAP)
                    if self.__LOW_MEMORY:
                        del fullImage
                        gc.collect()
                        fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

                ppMethod = pp.PostProcessingMethod(methodInfo['method'])
                dynargs = gather_dynamic_args(ppMethod)

                res = ppMethod.method(results=res, config=self.__CONFIG, args=methodInfo, dynargs=dynargs,
                                      display=not displayOnlyAP, verbose=verbose)

        """#######################
        ### Stats & Evaluation ###
        #######################"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if imageInfo["HAS_ANNOTATION"]:
                # TODO : Build automatically per_mask_conf_hierarchy
                per_mask_conf_hierarchy = None
                per_pixel_conf_hierarchy = None
                if self.__MODE == 'cortex':
                    per_pixel_conf_hierarchy = [2, {3: 1}]  # Ensures that cortex overrides other classes
                elif self.__MODE == 'main':
                    if len(self.__CUSTOM_CLASS_NAMES) == 10:
                        per_mask_conf_hierarchy = {3: [4, 5], 8: [9, 10], 10: [9]}
                    per_pixel_conf_hierarchy = [1, 2, {3: [4, 5]}, 6, 7, {8: [{10: 9}, 9]}]

                AP = self.__compute_ap_and_conf_matrix__(image_results_path, imageInfo, res, ground_truth,
                                                         per_mask_conf_hierarchy, per_pixel_conf_hierarchy,
                                                         save_results, displayOnlyAP)
                del ground_truth
            if not cortex_mode:
                self.__compute_statistics__(image_results_path, imageInfo, res, save_results, displayOnlyAP)
            if self.__MODE == "mest_glom" and (chainMode or (not chainMode and imageInfo['HAS_ANNOTATION'])):
                if chainMode:
                    base_results = self.__PREVIOUS_RES
                else:
                    base_results = self.__get_ground_truth__(image_results_path, fullImage, imageInfo,
                                                             displayOnlyAP, False, base_gt=True)
                stats.mask_histo_per_base_mask(
                    base_results=base_results,
                    results=res, image_info=imageInfo, classes={"nsg": "all"}, box_epsilon=0, config=self.__CONFIG,
                    test_masks=True, mask_threshold=0.9, display_per_base_mask=False, display_global=True,
                    save=image_results_path if save_results else None, verbose=verbose
                )

        """#################
        ### Finalization ###
        #################"""
        if save_results:
            if cortex_mode:
                self.__finalize_cortex_mode__(image_results_path, imageInfo, res, displayOnlyAP)
            if len(res['class_ids']) > 0:
                if not displayOnlyAP:
                    print(" - Applying masks on image")
                self.__STEP = "saving predicted image"
                self.__draw_masks__(image_results_path, fullImage, imageInfo, res,
                                    title=f"{imageInfo['NAME']} Predicted")
            elif not displayOnlyAP:
                print(" - No mask to apply on image")

            # Annotations Extraction
            if self.__CONFIG.has_to_export():
                self.__export_annotations__(image_results_path, imageInfo, res, displayOnlyAP)

        final_time = round(time() - start_time)
        print(f" Done in {formatTime(final_time)}\n")

        """########################################
        ### Extraction of needed classes' masks ###
        ########################################"""
        if chainMode:
            self.__PREVIOUS_RES = {}
            if self.__CONFIG.has_to_return():
                indices = np.arange(len(res['class_ids']))
                classIdsToGet = [self.__CLASS_2_ID[c] for c in self.__CONFIG.get_return_param()]
                indices = indices[np.isin(res['class_ids'], classIdsToGet)]
                for key in res:
                    if key == 'masks':
                        self.__PREVIOUS_RES[key] = res[key][..., indices]
                    else:
                        self.__PREVIOUS_RES[key] = res[key][indices, ...]

        if not imageInfo['HAS_ANNOTATION']:
            AP = -1
        self.__STEP = "finalizing"
        if save_results:
            with open(logsPath, 'a') as results_log:
                apText = f"{AP:4.3f}".replace(".", ",")
                results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText}%; {self.__MODEL_PATH};\n")
        del res, imageInfo, fullImage
        plt.clf()
        plt.close('all')
        gc.collect()
        return AP

    def __inference__(self, image, fullImage, imageInfo, allowSparse, displayOnlyAP):
        res = []
        total_px = self.__CONFIG.get_param()['roi_size'] ** 2
        skipped = 0
        skippedText = ""
        inference_start_time = time()
        if not displayOnlyAP:
            progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference')
        for divId in range(imageInfo["NB_DIV"]):
            self.__STEP = f"{divId} div processing"
            forceInference = False
            if 'X_STARTS' in imageInfo and 'Y_STARTS' in imageInfo:
                division = dD.getImageDivision(fullImage if image is None else image, imageInfo["X_STARTS"],
                                               imageInfo["Y_STARTS"], divId, self.__DIVISION_SIZE)
            elif 'ROI_COORDINATES' in imageInfo:
                currentRoI = imageInfo['ROI_COORDINATES'][divId]
                division = (fullImage if image is None else image)[currentRoI[0]:currentRoI[2],
                           currentRoI[1]:currentRoI[3], :]
                forceInference = True
            else:
                raise ValueError('Cannot find image areas to use')

            if not forceInference:
                grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
                colorPx = cv2.countNonZero(grayDivision)
                del grayDivision
            if forceInference or colorPx / total_px > 0.1:
                self.__STEP = f"{divId} div inference"
                results = self.__MODEL.process(division, normalizedCoordinates=False,
                                               score_threshold=self.__CONFIG.get_param()['min_confidence'])
                results["div_id"] = divId
                if self.__CONFIG.is_using_mini_mask():
                    res.append(utils.reduce_memory(results.copy(), config=self.__CONFIG, allow_sparse=allowSparse))
                else:
                    res.append(results.copy())
                del results
            elif not displayOnlyAP:
                skipped += 1
                skippedText = f"({skipped} empty division{'s' if skipped > 1 else ''} skipped) "
            del division
            gc.collect()

            if not displayOnlyAP:
                if divId + 1 == imageInfo["NB_DIV"]:
                    inference_duration = round(time() - inference_start_time)
                    skippedText += f"Duration = {formatTime(inference_duration)}"
                progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference', suffix=skippedText)
        if not displayOnlyAP:
            print(" - Fusing results of all divisions")
        self.__STEP = "fusing results"
        res = pp.fuse_results(res, imageInfo, division_size=self.__DIVISION_SIZE,
                              cortex_size=self.__RESIZE, config=self.__CONFIG)
        return res

    def __draw_masks__(self, image_results_path, img, image_info, masks_data, title=None, cleaned_image=True):
        if title is None:
            title = f"{image_info['NAME']} Masked"
        fileName = os.path.join(image_results_path, title.replace(' ', '_').replace('(', '').replace(')', ''))
        # No need of reloading or passing copy of image as it is the final drawing
        visualize.display_instances(
            img, masks_data['rois'], masks_data['masks'], masks_data['class_ids'],
            self.__VISUALIZE_NAMES, masks_data['scores'] if 'scores' in masks_data else None, colors=self.__COLORS,
            colorPerClass=True, fileName=fileName, save_cleaned_img=cleaned_image, silent=True, title=title, figsize=(
                (1024 if self.__MODE == "cortex" else image_info["WIDTH"]) / 100,
                (1024 if self.__MODE == "cortex" else image_info["HEIGHT"]) / 100
            ), image_format=image_info['IMAGE_FORMAT'], config=self.__CONFIG
        )

    def __get_ground_truth__(self, image_results_path, fullImage, image_info,
                             displayOnlyAP, save_results, base_gt=False):
        """
        Get ground-truth annotations and applies masks on image if enabled
        :param image_results_path: the current image output folder
        :param fullImage: the original image
        :param image_info: info about the current image
        :param displayOnlyAP: if True, will not print anything
        :param save_results: if True, will apply masks
        :return: gt_bbox, gt_class_id, gt_mask
        """
        datasetType = "base" if base_gt else "current"
        self.__STEP = f"{datasetType} dataset creation"
        if base_gt:
            config_ = self.__CONFIG.copy()
            config_.set_current_mode(self.__CONFIG.get_previous_mode())
        else:
            config_ = self.__CONFIG
        dataset_val = SkinetCustomDataset(image_info, config_, enable_occlusion=False)
        dataset_val.load_images()
        dataset_val.prepare()
        self.__STEP = f"loading {datasetType} annotated masks"
        image_id = dataset_val.image_ids[0]
        gt_mask, gt_class_id, gt_bbox = dataset_val.load_mask(image_id)
        gt = {'rois': gt_bbox, 'class_ids': gt_class_id, 'masks': gt_mask}
        if save_results:
            if len(gt_class_id) > 0:
                if not displayOnlyAP:
                    print(" - Applying annotations on file to get expected results")
                self.__STEP = "applying expected masks"
                self.__draw_masks__(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                                    image_info, gt, title=f"{image_info['NAME']}_Expected", cleaned_image=False)
            elif not displayOnlyAP:
                print(" - No mask expected, not saving expected image")
        return gt

    def __compute_statistics__(self, image_results_path, image_info, predicted, save_results=True, displayOnlyAP=False):
        """
        Computes area and counts masks of each class
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param save_results: Whether to save statistics in a file or not
        :return: None
        """
        self.__STEP = "computing statistics"
        _ = stats.get_count_and_area(predicted, image_info=image_info, selected_classes="all",
                                     save=image_results_path if save_results else None, display=True,
                                     config=self.__CONFIG)

    def __compute_ap_and_conf_matrix__(self, image_results_path, image_info, predicted, ground_truth,
                                       per_mask_conf_hierarchy, per_pixel_conf_hierarchy=None, save_results=True,
                                       displayOnlyAP=False):
        """
        Computes AP and confusion matrix
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param ground_truth: the ground truth results dictionary
        :param per_mask_conf_hierarchy: classes hierarchy for per mask confusion matrix
        :param per_pixel_conf_hierarchy: classes hierarchy for per pixel confusion matrix
        :param save_results: Whether to save files or not
        :param displayOnlyAP: Whether to display only AP or also current steps
        :return:
        """
        if not displayOnlyAP:
            print(" - Computing Average Precision and Confusion Matrix")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__STEP = "computing confusion matrix"
            conf_matrix = {}
            AP, _, _, _, conf_matrix['mask'] = utils.compute_ap(gt_boxes=ground_truth['rois'],
                                                                gt_class_ids=ground_truth['class_ids'],
                                                                gt_masks=ground_truth['masks'],
                                                                pred_boxes=predicted["rois"],
                                                                pred_class_ids=predicted["class_ids"],
                                                                pred_masks=predicted['masks'],
                                                                pred_scores=predicted["scores"],
                                                                nb_class=self.__NB_CLASS,
                                                                score_threshold=0.3,
                                                                iou_threshold=0.5,
                                                                confusion_iou_threshold=0.5,
                                                                classes_hierarchy=per_mask_conf_hierarchy,
                                                                confusion_background_class=True,
                                                                confusion_only_best_match=False)
            self.__APs.append(AP)
            self.__CONFUSION_MATRIX['mask'] = np.add(self.__CONFUSION_MATRIX['mask'], conf_matrix['mask'])
            if per_pixel_conf_hierarchy is None:
                per_pixel_conf_hierarchy = [i + 1 for i in range(self.__NB_CLASS)]
            if self.__RESIZE is not None:
                image_shape = self.__RESIZE
            else:
                image_shape = (image_info['HEIGHT'], image_info['WIDTH'])
            conf_matrix['pixel'] = utils.compute_confusion_matrix(
                image_shape=image_shape, config=self.__CONFIG,
                expectedResults=ground_truth, predictedResults=predicted,
                classes_hierarchy=per_pixel_conf_hierarchy, num_classes=self.__NB_CLASS
            )
            self.__CONFUSION_MATRIX['pixel'] = np.add(self.__CONFUSION_MATRIX['pixel'], conf_matrix['pixel'])
            print(f"{'' if displayOnlyAP else '   '} - Average Precision is about {AP:06.2%}")
            cmap = plt.cm.get_cmap('hot')
            if save_results:
                self.__STEP = "saving confusion matrix"
                for mat_type in ['mask', 'pixel']:
                    for normalized in [False, True]:
                        name = (f"{image_info['NAME']} Confusion Matrix ({mat_type.capitalize()})"
                                f"{' (Normalized)' if normalized else ''}")
                        confusionMatrixFileName = os.path.join(image_results_path, name.replace('(', '')
                                                               .replace(')', '')
                                                               .replace(' ', '_'))
                        visualize.display_confusion_matrix(conf_matrix[mat_type], self.__VISUALIZE_NAMES,
                                                           title=name, cmap=cmap, show=False, normalize=normalized,
                                                           fileName=confusionMatrixFileName)
        return AP

    def __finalize_cortex_mode__(self, image_results_path, image_info, predicted, displayOnlyAP):
        # TODO Generalize this step
        self.__STEP = "cleaning full resolution image"
        if not displayOnlyAP:
            print(" - Cleaning full resolution image and saving statistics")
        allCortices = None
        # Gathering every cortex masks into one
        for idxMask, classMask in enumerate(predicted['class_ids']):
            if classMask == 1:
                if allCortices is None:  # First mask found
                    allCortices = predicted['masks'][:, :, idxMask].copy() * 255
                else:  # Additional masks found
                    allCortices = cv2.bitwise_or(allCortices, predicted['masks'][:, :, idxMask] * 255)
        # To avoid cleaning an image without cortex
        if allCortices is not None:
            # Cleaning original image with cortex mask(s) and saving stats
            self.__keep_only_cortex__(image_results_path, image_info, allCortices)

    def __keep_only_cortex__(self, image_results_path, image_info, allCortices):
        """
        Cleans the original biopsy/nephrectomy with cortex masks and extracts the cortex area
        :param image_results_path: the current image output folder
        :param image_info: info about the current image
        :param allCortices: fused-cortices mask
        :return: allCorticesArea, allCorticesSmall
        """
        self.__STEP = "init fusion dir"
        fusion_dir = os.path.join(image_results_path, f"cortex")
        os.makedirs(fusion_dir, exist_ok=True)

        # Saving useful part of fused-cortices mask
        self.__STEP = "crop & save fused-cortices mask"
        allCorticesROI = utils.extract_bboxes(allCortices)
        allCorticesSmall = allCortices[allCorticesROI[0]:allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3]]
        cv2.imwrite(os.path.join(fusion_dir, f"{image_info['NAME']}_cortex.jpg"), allCorticesSmall, CV2_IMWRITE_PARAM)

        # Computing coordinates at full resolution
        yRatio = image_info['HEIGHT'] / self.__RESIZE[0]
        xRatio = image_info['WIDTH'] / self.__RESIZE[1]
        allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
        allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
        allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
        allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

        # Resizing fused-cortices mask and computing its area
        allCortices = cv2.resize(np.uint8(allCortices), (image_info['WIDTH'], image_info['HEIGHT']),
                                 interpolation=cv2.INTER_CUBIC)
        allCorticesArea = dD.getBWCount(allCortices)[1]
        stats = {
            "cortex": {
                "count": 1,
                "area": allCorticesArea,
                "x_offset": int(allCorticesROI[1]),
                "y_offset": int(allCorticesROI[0])
            }
        }
        with open(os.path.join(image_results_path, f"{image_info['NAME']}_stats.json"), "w") as saveFile:
            try:
                json.dump(stats, saveFile, indent='\t')
            except TypeError:
                print("    Failed to save statistics", flush=True)

        # Masking the image and saving it
        temp = np.repeat(allCortices[:, :, np.newaxis], 3, axis=2)
        image_info['ORIGINAL_IMAGE'] = cv2.bitwise_and(
            image_info['ORIGINAL_IMAGE'][allCorticesROI[0]: allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3], :],
            temp[allCorticesROI[0]: allCorticesROI[2], allCorticesROI[1]:allCorticesROI[3], :]
        )
        cv2.imwrite(os.path.join(fusion_dir, f"{image_info['NAME']}_cleaned.jpg"),
                    cv2.cvtColor(image_info['ORIGINAL_IMAGE'], cv2.COLOR_RGB2BGR),
                    CV2_IMWRITE_PARAM)
        return allCorticesArea, allCorticesSmall

    def __save_debug_image__(self, step, debugIterator, fullImage, image_info, res, image_results_path, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        title = f"{image_info['NAME']} Inference debug {debugIterator:02d} {step}"
        self.__draw_masks__(image_results_path, fullImage if self.__LOW_MEMORY else fullImage.copy(),
                            image_info, res, title, cleaned_image=False)

    def __export_annotations__(self, image_results_path, image_info, predicted, silent, verbose=0):
        """
        Exports predicted results as annotations files
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param silent: Whether to print text or not
        :return:
        """
        self.__STEP = "saving annotations"
        if not silent:
            print(" - Saving predicted annotations files")
        adapters = []
        export_param = self.__CONFIG.get_export_param()
        if type(export_param) is str:
            if export_param.lower() == "all":
                adapters = list(AnnotationAdapter.ANNOTATION_ADAPTERS.values())
            else:
                export_param = [export_param]
        if type(export_param) is list:
            for adapterName in export_param:
                adapter = AnnotationAdapter.getAdapterFromName(adapterName)
                if adapter is not None:
                    adapters.append(adapter)
        for adapter in adapters:
            try:
                # TODO Fix export
                export_annotations(image_info, predicted, adapter, save_path=image_results_path,
                                   config=self.__CONFIG, verbose=verbose if silent else 1)
            except Exception:
                print(f"Failed to export using {adapter.getName()} adapter")
