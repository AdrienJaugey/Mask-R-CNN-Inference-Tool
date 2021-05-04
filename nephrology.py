import json
import os
import re
import traceback
import shutil
import sys
import warnings
import gc

from datasetTools.CustomDataset import SkinetCustomDataset

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        with open(configPath, 'r') as jsonFile:
            self.__CONFIG_FILE = json.load(jsonFile)
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

        self.__CONFIG = None

        self.__PREVIOUS_RES = None

    def __get_mode_config__(self, mode: str = None):
        return self.__CONFIG_FILE['modes'][self.__MODE if mode is None else mode]

    def load(self, mode: str, forceFullSizeMasks=False, forceModelPath=None):
        # Loading the inference mode
        mode_config = self.__get_mode_config__(mode)

        # If mode is already loaded, nothing to do
        if self.__MODE == mode:
            self.__CONFIG.USE_MINI_MASK = not forceFullSizeMasks and mode_config['parameters']['mini_mask'] is not None
            self.__CONFIG.MINI_MASK_SHAPE = tuple([mode_config['parameters']['mini_mask']
                                                   if self.__CONFIG.USE_MINI_MASK else 0] * 2)  # (H, W)
            print(f"{mode} mode is already loaded.\n")
            return

        self.__MODEL_PATH = find_latest_weight(mode_config['parameters']['weight_file']
                                               if forceModelPath is None else forceModelPath)

        # Testing only for one of the format, as Error would have already been raised if modelPath was not correct
        isExportedModelDir = os.path.exists(os.path.join(self.__MODEL_PATH, 'saved_model'))
        if isExportedModelDir:
            self.__MODEL_PATH = os.path.join(self.__MODEL_PATH, 'saved_model')

        self.__CLASS_2_ID = {}
        self.__ID_2_CLASS = {}
        for idx, classInfo in enumerate(mode_config['classes']):
            _idx = idx + 1
            classInfo['id'] = _idx
            self.__ID_2_CLASS[_idx] = classInfo['name']
            self.__CLASS_2_ID[classInfo['name']] = _idx
        self.__CLASSES_INFO = mode_config['classes'].copy()

        if self.__MODEL is None:
            self.__MODEL = TensorflowDetector(self.__MODEL_PATH, {c['id']: c for c in mode_config['classes']})
        else:
            self.__MODEL.load(self.__MODEL_PATH, {c['id']: c for c in mode_config['classes']})
        self.__READY = self.__MODEL.isLoaded()
        if not self.__MODEL.isLoaded():
            raise ValueError("Please provide correct path to model.")

        self.__MODE = mode

        self.__DIVISION_SIZE = mode_config['parameters']['roi_size']
        self.__MIN_OVERLAP_PART = mode_config['parameters']['min_overlap_part']
        if mode_config['parameters']['resize'] is None:
            self.__RESIZE = None
        else:
            self.__RESIZE = tuple(mode_config['parameters']['resize'])

        self.__NB_CLASS = len(mode_config['classes'])
        self.__CUSTOM_CLASS_NAMES = [classInfo["name"] for classInfo in mode_config['classes']]
        self.__VISUALIZE_NAMES = ['Background']
        self.__VISUALIZE_NAMES.extend([classInfo.get('display_name', classInfo['name'])
                                       for classInfo in mode_config['classes']])

        self.__COLORS = [classInfo["color"] for classInfo in mode_config['classes']]

        # Configurations
        self.__CONFUSION_MATRIX = {'pixel': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64),
                                   'mask': np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int64)}
        self.__APs = []

        class SkinetConfig:
            NAME = "skinet"
            NUM_CLASSES = 1 + len(mode_config['classes'])
            IMAGE_SIDE = mode_config['parameters']['roi_size']
            MIN_OVERLAP = mode_config['parameters']['min_overlap_part']
            DETECTION_MIN_CONFIDENCE = mode_config['parameters']['min_confidence']
            USE_MINI_MASK = not forceFullSizeMasks and mode_config['parameters']['mini_mask'] is not None
            MINI_MASK_SHAPE = tuple([mode_config['parameters']['mini_mask'] if USE_MINI_MASK else 0] * 2)  # (H, W)

        self.__CONFIG = SkinetConfig()
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
        mode_config = self.__get_mode_config__()
        parameters = mode_config['parameters']
        roi_mode = parameters["roi_mode"]
        suffix = ""
        if chainMode and mode_config['previous'] is not None:
            image_name, extension = os.path.splitext(os.path.basename(imagePath))
            extension = extension.replace('.', '')
            if extension not in ['png', 'jpg']:
                extension = 'jpg'
            if self.__get_mode_config__(mode_config['previous'])['parameters'].get('resize', None) is not None:
                suffix = "_base"
            imagePath = os.path.join(results_path, image_name, mode_config['previous'],
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

            if parameters.get('base_class', None) is not None:
                imageInfo['BASE_CLASS'] = parameters['base_class']

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
            if parameters['allow_empty_annotations']:
                imageInfo['HAS_ANNOTATION'] = annotationExists

            previous_mode = self.__get_mode_config__().get('previous', None)
            if chainMode and 'BASE_CLASS' in imageInfo and previous_mode is not None:
                previous_mode = self.__get_mode_config__(previous_mode)
                baseClassId = 0
                for idx, aClass in enumerate(previous_mode['classes']):
                    if aClass['name'] == imageInfo['BASE_CLASS']:
                        baseClassId = idx + 1
                        break
                if self.__PREVIOUS_RES is not None and 'class_ids' in self.__PREVIOUS_RES \
                        and baseClassId in self.__PREVIOUS_RES['class_ids']:
                    indices = np.arange(len(self.__PREVIOUS_RES['class_ids']))
                    indices = indices[np.isin(self.__PREVIOUS_RES['class_ids'], [baseClassId])]
                    if len(indices) > 0:
                        temp = previous_mode['parameters'].get("mini_mask", None)
                        previousModeUsedMiniMask = temp is not None and self.__PREVIOUS_RES['masks'].shape[0] == temp\
                                                   and self.__PREVIOUS_RES['masks'].shape[1] == temp
                        previousResize = previous_mode['parameters'].get('resize', None)
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

                        crop_to_base_class = mode_config['parameters'].get('crop_to_base_class', False)
                        if crop_to_base_class:
                            fusedBbox = utils.extract_bboxes(fusedMask)
                            image = image[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                            fullImage = fullImage[fusedBbox[0]:fusedBbox[2], fusedBbox[1]:fusedBbox[3], :]
                            imsave(imageInfo['PATH'], fullImage)
                            height, width = image.shape[:2]
                            offset = np.array([fusedBbox[0], fusedBbox[1]] * 2)

                        # If RoI mode is 'centered', inference will be done on base-class masks
                        if roi_mode == 'centered':
                            if parameters['fuse_base_class']:
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
                        if parameters['fuse_base_class']:
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
                                if not parameters['fuse_base_class']:
                                    imageInfo['BASE_COUNT'] += 1
                                del mask
                        imageInfo['HEIGHT'] = int(height)
                        imageInfo['WIDTH'] = int(width)
            elif annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible",
                          flush=True)
                dW.createMasksOfImage(imageInfo['DIR_PATH'], imageInfo['NAME'], 'data', classesInfo=self.__CLASSES_INFO,
                                      imageFormat=imageInfo['IMAGE_FORMAT'], resize=self.__RESIZE, config=self.__CONFIG)
                maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                if 'BASE_CLASS' in imageInfo and imageInfo['BASE_CLASS'] in maskDirs:
                    # Fusing masks of base class if needed, then cleaning image using it/them
                    if parameters['fuse_base_class']:
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

    def __init_results_dir__(self, results_path, chainMode=False):
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
            results_path, logsPath = self.__init_results_dir__(results_path, chainMode=chainMode)
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
                    nextMode = self.__CONFIG_FILE['first_mode']
                    while nextMode is not None:
                        print(f"[{nextMode} mode]")
                        self.load(nextMode, forceFullSizeMasks=chainModeForceFullSize)
                        nextMode = self.__get_mode_config__().get("next", None)
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
        allowSparse = self.__get_mode_config__()['parameters'].get('allow_sparse', True)
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

        """##############
        ### Inference ###
        ##############"""
        res = []
        total_px = self.__CONFIG.IMAGE_SIDE ** 2
        skipped = 0
        debugIterator = -1
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
                                               score_threshold=self.__CONFIG.DETECTION_MIN_CONFIDENCE)
                results["div_id"] = divId
                if self.__CONFIG.USE_MINI_MASK:
                    res.append(utils.reduce_memory(results.copy(), config=self.__CONFIG,
                                                   allow_sparse=allowSparse))
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

        """####################
        ### Post-Processing ###
        ####################"""
        if not displayOnlyAP:
            print(" - Fusing results of all divisions")
        self.__STEP = "fusing results"
        res = pp.fuse_results(res, imageInfo, division_size=self.__DIVISION_SIZE,
                              cortex_size=self.__RESIZE, config=self.__CONFIG)

        if len(res['class_ids']) > 0:
            mode_config = self.__get_mode_config__()
            for idx, methodInfo in enumerate(mode_config['post_processing']):
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
                dynargs = {}
                for dynarg in ppMethod.dynargs():
                    if dynarg == 'image':
                        dynargs[dynarg] = image
                    elif dynarg == 'classes_info':
                        dynargs[dynarg] = self.__CLASSES_INFO.copy()
                    elif dynarg == 'image_info':
                        dynargs[dynarg] = imageInfo.copy()
                    else:
                        raise NotImplementedError(f'Dynamic argument \'{dynarg}\' is not implemented.')
                if methodInfo['method'] not in ['statistics', 'export_as_annotations']:
                    res = ppMethod.method(results=res, args=methodInfo, config=self.__CONFIG,
                                          display=not displayOnlyAP, verbose=verbose, dynargs=dynargs)
                else:
                    raise NotImplementedError(f"Dynamic post-processing method \'{methodInfo['method']}\' is not "
                                              f"implemented.")

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
                self.__compute_statistics__(image_results_path, imageInfo, res, save_results)
        if save_results:
            if cortex_mode:
                self.__finalize_cortex_mode__(image_results_path, imageInfo, res, displayOnlyAP)
            if len(res['class_ids']) > 0:
                if not displayOnlyAP:
                    print(" - Applying masks on image")
                self.__STEP = "saving predicted image"
                self.__draw_masks__(image_results_path, fullImage, imageInfo, res, title=f"{imageInfo['NAME']} Predicted")
            elif not displayOnlyAP:
                print(" - No mask to apply on image")

            # Annotations Extraction
            self.__export_annotations__(image_results_path, imageInfo, res, displayOnlyAP)

        final_time = round(time() - start_time)
        print(f" Done in {formatTime(final_time)}\n")

        """########################################
        ### Extraction of needed classes' masks ###
        ########################################"""
        if chainMode:
            self.__PREVIOUS_RES = {}
            if self.__get_mode_config__().get('return', None) is not None:
                indices = np.arange(len(res['class_ids']))
                classIdsToGet = [self.__CLASS_2_ID[c] for c in self.__get_mode_config__()['return']]
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
                results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText}%; {self.__MODE};\n")
        del res, imageInfo, fullImage
        plt.clf()
        plt.close('all')
        gc.collect()
        return AP

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

    def __get_ground_truth__(self, image_results_path, fullImage, image_info, displayOnlyAP, save_results):
        """
        Get ground-truth annotations and applies masks on image if enabled
        :param image_results_path: the current image output folder
        :param fullImage: the original image
        :param image_info: info about the current image
        :param displayOnlyAP: if True, will not print anything
        :param save_results: if True, will apply masks
        :return: gt_bbox, gt_class_id, gt_mask
        """
        self.__STEP = "dataset creation"
        dataset_val = SkinetCustomDataset(self.__CUSTOM_CLASS_NAMES, self.__MODE, self.__RESIZE,
                                          self.__CONFIG, image_info, enable_occlusion=False)
        dataset_val.load_images()
        dataset_val.prepare()
        self.__STEP = "loading annotated masks"
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

    def __compute_statistics__(self, image_results_path, image_info, predicted, save_results=True):
        """
        Computes area and counts masks of each class
        :param image_results_path: output folder of current image
        :param image_info: info about current image
        :param predicted: the predicted results dictionary
        :param save_results: Whether to save statistics in a file or not
        :return: None
        """
        self.__STEP = "computing statistics"
        print(" - Computing statistics on predictions")
        stats = pp.get_count_and_area(predicted, classes_info=self.__CLASSES_INFO,
                                      selected_classes=self.__CUSTOM_CLASS_NAMES, config=self.__CONFIG)
        if 'BASE_CLASS' in image_info:
            if 'BASE_CLASS' in image_info:
                stats[image_info['BASE_CLASS']] = {"count": image_info['BASE_COUNT'],
                                                   "area": image_info["BASE_AREA"]}
        for className in stats:
            stat = stats[className]
            print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")
        if save_results:
            with open(os.path.join(image_results_path, f"{image_info['NAME']}_stats.json"), "w") as saveFile:
                try:
                    json.dump(stats, saveFile, indent='\t')
                except TypeError:
                    print("    Failed to save statistics", flush=True)

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
            allCorticesArea, allCorticesSmall = self.__keep_only_cortex__(
                image_results_path, image_info, allCortices
            )

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
        for adapter in AnnotationAdapter.ANNOTATION_ADAPTERS:
            try:
                pp.export_annotations(image_info, predicted, self.__CLASSES_INFO,
                                      adapter, save_path=image_results_path,
                                      config=self.__CONFIG, verbose=verbose if silent else 1)
            except Exception:
                print(f"Failed to export using {adapter.__qualname__} adapter")
