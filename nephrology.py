import json
import os
import re
import traceback
import shutil
import sys
import warnings
import gc

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
    from datasetTools import datasetDivider as dD, datasetWrapper as dW, AnnotationAdapter

    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
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

    def __init__(self, classesInfo, modelPath, min_confidence=0.5, divisionSize=1024, min_overlap_part_main=0.33,
                 min_overlap_part_cortex=0.5, cortex_size=None, mini_mask_size=96, forceFullSizeMasks=False,
                 low_memory=False):
        print("Initialisation")
        self.__CLASSES_INFO = classesInfo
        self.__CORTEX_MODE = not classesInfo[0]["ignore"]
        cortex_mode = self.__CORTEX_MODE
        self.__MODEL_PATH = find_latest_weight(modelPath)
        self.__DIVISION_SIZE = divisionSize
        self.__MIN_OVERLAP_PART_MAIN = min_overlap_part_main
        self.__MIN_OVERLAP_PART_CORTEX = min_overlap_part_cortex
        self.__MIN_OVERLAP_PART = min_overlap_part_cortex if self.__CORTEX_MODE else min_overlap_part_main
        self.__CORTEX_SIZE = None if not self.__CORTEX_MODE else (1024, 1024) if cortex_size is None else cortex_size
        self.__LOW_MEMORY = low_memory
        self.__CUSTOM_CLASS_NAMES = []
        for classInfo in classesInfo:
            if not classInfo["ignore"]:
                self.__CUSTOM_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CUSTOM_CLASS_NAMES)
        # Root directory of the project
        self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        self.__CONFUSION_MATRIX = np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int32)
        self.__APs = []

        class SkinetConfig(Config):
            NAME = "skinet"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + nbClass
            IMAGE_MIN_DIM = divSize
            IMAGE_MAX_DIM = divSize
            RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)
            TRAIN_ROIS_PER_IMAGE = 800
            DETECTION_MIN_CONFIDENCE = min_confidence
            STEPS_PER_EPOCH = 400
            VALIDATION_STEPS = 50
            USE_MINI_MASK = not cortex_mode and not forceFullSizeMasks
            MINI_MASK_SHAPE = (mini_mask_size, mini_mask_size)  # (height, width) of the mini-mask

        self.__CONFIG = SkinetConfig()

        # Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=self.__CONFIG, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        assert self.__MODEL_PATH is not None and self.__MODEL_PATH != "", "Provide path to trained weights"
        print("Loading weights from", self.__MODEL_PATH)
        self.__MODEL.load_weights(self.__MODEL_PATH, by_name=True)
        print()

    def prepare_image(self, imagePath, results_path, silent=False):
        """
        Creating png version if not existing, dataset masks if annotation found and get some information
        :param imagePath: path to the image to use
        :param results_path: path to the results dir to create the image folder and paste it in
        :param silent: No display
        :return: image, imageInfo = {"PATH": str, "DIR_PATH": str, "FILE_NAME": str, "NAME": str, "HEIGHT": int,
        "WIDTH": int, "NB_DIV": int, "X_STARTS": v, "Y_STARTS": list, "HAS_ANNOTATION": bool}
        """
        image = None
        fullImage = None
        imageInfo = None
        image_results_path = None
        if os.path.exists(imagePath):
            imageInfo = {
                'PATH': imagePath,
                'DIR_PATH': os.path.dirname(imagePath),
                'FILE_NAME': os.path.basename(imagePath)
            }
            imageInfo['NAME'] = imageInfo['FILE_NAME'].split('.')[0]
            imageInfo['IMAGE_FORMAT'] = imageInfo['FILE_NAME'].split('.')[-1]

            # Reading input image in RGB color order
            imageChanged = False
            if self.__CORTEX_MODE:  # If in cortex mode, resize image to lower resolution
                imageInfo['FULL_RES_IMAGE'] = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                fullImage = cv2.resize(imageInfo['FULL_RES_IMAGE'], self.__CORTEX_SIZE)
                imageChanged = True
            else:
                fullImage = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
                height, width, _ = fullImage.shape
            imageInfo['HEIGHT'] = int(height)
            imageInfo['WIDTH'] = int(width)
            imageInfo['CORTEX_AREA'] = height * width

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
                os.makedirs(image_results_path, exist_ok=True)
                imageInfo['PATH'] = os.path.join(image_results_path, f"{imageInfo['NAME']}.{imageInfo['IMAGE_FORMAT']}")
                if not imageChanged:
                    shutil.copy2(imagePath, imageInfo['PATH'])
                else:
                    imsave(imageInfo['PATH'], fullImage)
            else:
                image_results_path = None

            # Computing divisions coordinates if needed and total number of div
            if self.__DIVISION_SIZE == "noDiv":
                imageInfo['X_STARTS'] = imageInfo['Y_STARTS'] = [0]
            else:
                imageInfo['X_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[0] if self.__CORTEX_MODE else width,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
                imageInfo['Y_STARTS'] = dD.computeStartsOfInterval(
                    maxVal=self.__CORTEX_SIZE[1] if self.__CORTEX_MODE else height,
                    intervalLength=self.__DIVISION_SIZE,
                    min_overlap_part=self.__MIN_OVERLAP_PART
                )
            imageInfo['NB_DIV'] = dD.getDivisionsCount(imageInfo['X_STARTS'], imageInfo['Y_STARTS'])

            # If annotations found, create masks and clean image if possible
            annotationExists = False
            imageInfo['HAS_ANNOTATION'] = False
            for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                annotationExists = annotationExists or os.path.exists(os.path.join(imageInfo['DIR_PATH'],
                                                                                   imageInfo['NAME'] + '.' + ext))
            if annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible")
                dW.createMasksOfImage(imageInfo['DIR_PATH'], imageInfo['NAME'], 'data', classesInfo=self.__CLASSES_INFO,
                                      imageFormat=imageInfo['IMAGE_FORMAT'], resize=self.__CORTEX_SIZE,
                                      config=self.__CONFIG)
                if not self.__CORTEX_MODE:
                    dW.fuseCortices('data', imageInfo['NAME'], imageFormat=imageInfo['IMAGE_FORMAT'],
                                    deleteBaseMasks=True, silent=True)
                    dW.cleanImage('data', imageInfo['NAME'], onlyMasks=False)
                maskDirs = os.listdir(os.path.join('data', imageInfo['NAME']))
                if not self.__CORTEX_MODE:
                    if "cortex" in maskDirs:
                        cortexDirPath = os.path.join('data', imageInfo['NAME'], 'cortex')
                        cortexImageFilePath = os.listdir(cortexDirPath)[0]
                        cortex = dW.loadSameResImage(os.path.join(cortexDirPath, cortexImageFilePath), fullImage.shape)
                        imageInfo['CORTEX_AREA'] = dD.getBWCount(cortex)[1]
                        del cortex
                    # If full_images directory exists it means than image has been cleaned so we have to get it another
                    # time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', imageInfo['NAME'], 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        imageInfo['cleaned_image_path'] = os.path.join(imagesDirPath, imageFilePath)
                        image = cv2.cvtColor(cv2.imread(imageInfo['cleaned_image_path']), cv2.COLOR_BGR2RGB)

                # We want to know if image has annotation, if we don't want to detect cortex and this mask exist
                # as we need it to clean the image, we remove it from the mask list before checking if a class
                # we want to predict has an annotated mask
                if "cortex" not in self.__CUSTOM_CLASS_NAMES and "cortex" in maskDirs:
                    maskDirs.remove("cortex")
                for classToPredict in self.__CUSTOM_CLASS_NAMES:
                    if classToPredict in maskDirs:
                        imageInfo['HAS_ANNOTATION'] = True
                        break
                if imageInfo['HAS_ANNOTATION'] and not silent:
                    print("    - AP and confusion matrix will be computed")

        return image, fullImage, imageInfo, image_results_path

    def init_results_dir(self, results_path):
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
            results_log.write(f"Image; Duration (s); Precision; {os.path.basename(self.__MODEL_PATH)}\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, save_results=True,
                  fusion_bb_threshold=0., fusion_mask_threshold=0.1,
                  filter_bb_threshold=0.5, filter_mask_threshold=0.9,
                  priority_table=None, nbMaxDivPerAxis=3, fusionDivThreshold=0.1,
                  displayOnlyAP=False, savePreFusionImage=False, savePreFilterImage=False,
                  allowSparse=True, minMaskArea=300, on_border_threshold=0.25):

        if len(images) == 0:
            print("Images list is empty, no inference to perform.")
            return

        # If results have to be saved, setting the results path and creating directory
        if save_results:
            results_path, logsPath = self.init_results_dir(results_path)
        else:
            print("No result will be saved")
            results_path = None

        self.__CONFUSION_MATRIX.fill(0)
        self.__APs.clear()
        cmap = plt.cm.get_cmap('hot')
        total_start_time = time()
        failedImages = []
        for i, IMAGE_PATH in enumerate(images):
            try:
                # Last step of full image inference
                if '_fusion_info.skinet' in IMAGE_PATH:
                    if self.__CORTEX_MODE:
                        continue
                    step = "initialisation"
                    with open(IMAGE_PATH, 'r') as skinetFile:
                        fusionInfo = json.load(skinetFile)
                    print(f"Finalising {fusionInfo['image']} image {progressText(i + 1, len(images))}")
                    fusionDir = fusionInfo["image"] + '_fusion'
                    fusionDirPath = os.path.join(os.path.dirname(IMAGE_PATH), fusionDir)
                    # TODO Dynamic image extension !!!
                    cleaned_image_path = os.path.join(fusionDirPath, fusionInfo["image"] + "_cleaned.jpg")
                    cleanedImage = cv2.imread(cleaned_image_path)
                    image_results_path = os.path.join(results_path, fusionInfo['image'])
                    os.makedirs(image_results_path, exist_ok=True)

                    # Pasting each divisions and merging all the stats
                    globalStats = None
                    for imageFolder in os.listdir(results_path):
                        step = f"{imageFolder} div fusion"
                        if (fusionInfo['image'] + '_') in imageFolder:
                            divID = imageFolder.split('_')[-1]
                            if fusionInfo["divisions"][divID]["used"]:
                                imagePath = os.path.join(results_path, imageFolder,
                                                         imageFolder + "_predicted_clean.png")
                                statsPath = os.path.join(results_path, imageFolder, imageFolder + "_stats.json")
                                with open(statsPath, 'r') as tempStatsFile:
                                    tempStats = json.load(tempStatsFile)
                                if "cortex" in tempStats:
                                    tempStats["cortex"]["area"] = fusionInfo["divisions"][divID]["cortex_area"]
                                    with open(statsPath, 'w') as tempStatsFile:
                                        json.dump(tempStats, tempStatsFile, indent="\t")
                                if globalStats is None:
                                    globalStats = tempStats
                                else:
                                    for className in tempStats:
                                        if className == "cortex":
                                            continue
                                        globalStats[className]["count"] += tempStats[className]["count"]
                                        globalStats[className]["area"] += tempStats[className]["area"]
                                divImage = cv2.imread(imagePath)
                                coo = fusionInfo["divisions"][divID]["coordinates"]
                                cleanedImage[coo["y1"]:coo["y2"], coo["x1"]:coo["x2"], :] = divImage
                                shutil.move(os.path.join(results_path, imageFolder),
                                            os.path.join(image_results_path, "divisions", imageFolder))
                    cv2.imwrite(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction.jpg"),
                                cleanedImage, CV2_IMWRITE_PARAM)
                    with open(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction_stats.json"),
                              'w') as globalStatsFile:
                        globalStats["cortex"] = {"count": 1, "area": fusionInfo["cortex_area"]}
                        json.dump(globalStats, globalStatsFile, indent="\t")
                    del cleanedImage
                    print("Done\n")
                else:
                    start_time = time()
                    print(f"Using {IMAGE_PATH} image file {progressText(i + 1, len(images))}")
                    visualizeNames = self.__CUSTOM_CLASS_NAMES.copy()
                    visualizeNames.insert(0, 'background')

                    step = "image preparation"
                    image, fullImage, imageInfo, image_results_path = self.prepare_image(IMAGE_PATH, results_path,
                                                                                         silent=displayOnlyAP)
                    if imageInfo["HAS_ANNOTATION"]:
                        step = "dataset creation"

                        class CustomDataset(utils.Dataset):

                            def __init__(self, custom_class_names, cortex_mode, cortex_size, config, image_format):
                                super().__init__()
                                self.__CUSTOM_CLASS_NAMES = custom_class_names.copy()
                                self.__CORTEX_MODE = cortex_mode
                                self.__CORTEX_SIZE = cortex_size
                                self.__CONFIG = config
                                self.__IMAGE_FORMAT = image_format

                            def get_class_names(self):
                                return self.__CUSTOM_CLASS_NAMES.copy()

                            def load_images(self):
                                # Add classes
                                for class_id, class_name in enumerate(self.__CUSTOM_CLASS_NAMES):
                                    self.add_class("skinet", class_id + 1, class_name)
                                image_name = imageInfo["NAME"]
                                img_path = os.path.join('data', image_name, "images",
                                                        f"{image_name}.{self.__IMAGE_FORMAT}")
                                self.add_image("skinet", image_id=imageInfo["NAME"], path=img_path)

                            def image_reference(self, image_id_):
                                """Return the skinet data of the image."""
                                info = self.image_info[image_id_]
                                if info["source"] == "skinet":
                                    return info["skinet"]
                                else:
                                    super(self.__class__).image_reference(self, image_id_)

                            def load_mask(self, image_id_):
                                """Generate instance masks for cells of the given image ID.
                                """
                                info = self.image_info[image_id_]
                                info = info.get("id")

                                path = os.path.join('data', info)

                                # Counting masks for current image
                                number_of_masks = 0
                                for masks_dir in os.listdir(path):
                                    # For each directory excepting /images
                                    if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                                        continue
                                    temp_DIR = os.path.join(path, masks_dir)
                                    # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
                                    number_of_masks += len([name_ for name_ in os.listdir(temp_DIR)
                                                            if os.path.isfile(os.path.join(temp_DIR, name_))])
                                if self.__CORTEX_MODE:
                                    masks_shape = (self.__CORTEX_SIZE[0], self.__CORTEX_SIZE[1], number_of_masks)
                                elif self.__CONFIG.USE_MINI_MASK:
                                    masks_shape = (self.__CONFIG.MINI_MASK_SHAPE[0], self.__CONFIG.MINI_MASK_SHAPE[1],
                                                   number_of_masks)
                                else:
                                    masks_shape = (imageInfo["HEIGHT"], imageInfo["WIDTH"], number_of_masks)
                                masks = np.zeros(masks_shape, dtype=np.uint8)
                                bboxes = np.zeros((number_of_masks, 4), dtype=np.int32)
                                iterator = 0
                                class_ids = np.zeros((number_of_masks,), dtype=int)
                                for masks_dir in os.listdir(path):
                                    if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                                        continue
                                    temp_class_id = self.__CUSTOM_CLASS_NAMES.index(masks_dir) + 1
                                    masks_dir_path = os.path.join(path, masks_dir)
                                    for mask_file in os.listdir(masks_dir_path):
                                        mask = imread(os.path.join(masks_dir_path, mask_file))
                                        masks[:, :, iterator] = mask
                                        if self.__CONFIG.USE_MINI_MASK:
                                            bboxes[iterator] = dW.getBboxFromName(mask_file)
                                        else:
                                            bboxes[iterator] = utils.extract_bboxes(mask)
                                        class_ids[iterator] = temp_class_id
                                        iterator += 1
                                # Handle occlusions /!\ In our case there is no possible occlusion (part of object that
                                # is hidden), all objects are complete (some are parts of other)
                                # occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
                                # for i in range(number_of_masks - 2, -1, -1):
                                #     masks[:, :, i] = masks[:, :, i] * occlusion
                                #     occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
                                return masks, class_ids.astype(np.int32), bboxes

                        dataset_val = CustomDataset(self.__CUSTOM_CLASS_NAMES, self.__CORTEX_MODE,
                                                    self.__CORTEX_SIZE, self.__CONFIG, imageInfo["IMAGE_FORMAT"])
                        dataset_val.load_images()
                        dataset_val.prepare()

                        step = "loading annotated masks"
                        image_id = dataset_val.image_ids[0]
                        gt_mask, gt_class_id, gt_bbox = dataset_val.load_mask(image_id)
                        if save_results:
                            if not displayOnlyAP:
                                print(" - Applying annotations on file to get expected results")
                            fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Expected")
                            _ = visualize.display_instances(fullImage if self.__LOW_MEMORY else fullImage.copy(),
                                                            gt_bbox, gt_mask, gt_class_id, visualizeNames,
                                                            colorPerClass=True, figsize=(
                                                             (1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                             (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100
                                                            ), image_format=imageInfo['IMAGE_FORMAT'],
                                                            title=f"{imageInfo['NAME']} Expected",
                                                            fileName=fileName, silent=True, config=self.__CONFIG)
                            if self.__LOW_MEMORY:
                                del fullImage
                                gc.collect()
                                fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)

                    # Getting predictions for each division

                    res = []
                    total_px = self.__CONFIG.IMAGE_MAX_DIM * self.__CONFIG.IMAGE_MIN_DIM
                    skipped = 0
                    debugIterator = -1
                    skippedText = ""
                    inference_start_time = time()
                    if not displayOnlyAP:
                        progressBar(0, imageInfo["NB_DIV"], prefix=' - Inference ')
                    for divId in range(imageInfo["NB_DIV"]):
                        step = f"{divId} div processing"
                        division = dD.getImageDivision(fullImage if image is None else image, imageInfo["X_STARTS"],
                                                       imageInfo["Y_STARTS"], divId)
                        grayDivision = cv2.cvtColor(division, cv2.COLOR_RGB2GRAY)
                        colorPx = cv2.countNonZero(grayDivision)
                        del grayDivision
                        if colorPx / total_px > 0.1:
                            step = f"{divId} div inference"
                            results = self.__MODEL.detect([division])
                            results[0]["div_id"] = divId
                            if self.__CONFIG.USE_MINI_MASK:
                                res.append(utils.reduce_memory(results[0].copy(), config=self.__CONFIG,
                                                               allow_sparse=allowSparse))
                            else:
                                res.append(results[0].copy())
                            del results
                        elif not displayOnlyAP:
                            skipped += 1
                            skippedText = f" ({skipped} empty division{'s' if skipped > 1 else ''} skipped)"
                        del division
                        gc.collect()
                        if not displayOnlyAP:
                            if divId + 1 == imageInfo["NB_DIV"]:
                                inference_duration = round(time() - inference_start_time)
                                skippedText += f" Duration = {formatTime(inference_duration)}"
                            progressBar(divId + 1, imageInfo["NB_DIV"], prefix=' - Inference ', suffix=skippedText)

                    # Post-processing of the predictions
                    if not displayOnlyAP:
                        print(" - Fusing results of all divisions")

                    step = "fusing results"
                    res = pp.fuse_results(res, fullImage.shape, division_size=self.__DIVISION_SIZE,
                                          min_overlap_part=self.__MIN_OVERLAP_PART)

                    if savePreFusionImage:
                        debugIterator += 1
                        self.save_debug_image("pre fusion", debugIterator, fullImage, imageInfo, res,
                                              image_results_path, visualizeNames, silent=displayOnlyAP)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        step = "fusing masks"
                        progressBarPrefix = " - Fusing overlapping masks " if not displayOnlyAP else None
                        res = pp.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                            config=self.__CONFIG, displayProgress=progressBarPrefix, verbose=0)

                        if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                            if savePreFilterImage:
                                debugIterator += 1
                                self.save_debug_image("pre border filter", debugIterator, fullImage, imageInfo, res,
                                                      image_results_path, visualizeNames, silent=displayOnlyAP)
                            step = "removing border masks"
                            progressBarPrefix = " - Removing border masks " if not displayOnlyAP else None
                            classes_to_check = [7, 8, 9, 10]
                            res = pp.filter_on_border_masks(res, fullImage if image is None else image,
                                                            onBorderThreshold=on_border_threshold,
                                                            classes=classes_to_check, config=self.__CONFIG,
                                                            displayProgress=progressBarPrefix, verbose=0)

                            if savePreFilterImage:
                                debugIterator += 1
                                self.save_debug_image("pre orphan filter", debugIterator, fullImage, imageInfo, res,
                                                      image_results_path, visualizeNames, silent=displayOnlyAP)
                                # TODO : Build automatically classes_hierarchy

                            classes_hierarchy = {
                                3: {"contains": [4, 5], "keep_if_no_child": False},
                                8: {"contains": [9, 10], "keep_if_no_child": True}
                            }
                            step = "filtering orphan masks (pass 1)"
                            progressBarPrefix = " - Removing orphan masks " if not displayOnlyAP else None
                            res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                         mask_threshold=filter_mask_threshold,
                                                         classes_hierarchy=classes_hierarchy,
                                                         displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                         verbose=0)
                        del image
                        if savePreFilterImage:
                            debugIterator += 1
                            self.save_debug_image("pre filter", debugIterator, fullImage, imageInfo, res,
                                                  image_results_path, visualizeNames, silent=displayOnlyAP)

                        step = "filtering masks"
                        progressBarPrefix = " - Removing non-sense masks " if not displayOnlyAP else None
                        res = pp.filter_masks(res, bb_threshold=filter_bb_threshold, priority_table=priority_table,
                                              mask_threshold=filter_mask_threshold, included_threshold=0.9,
                                              including_threshold=0.6, verbose=0,
                                              displayProgress=progressBarPrefix, config=self.__CONFIG)

                        if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                            if savePreFilterImage:
                                debugIterator += 1
                                self.save_debug_image("pre orphan filter (Pass 2)", debugIterator, fullImage, imageInfo,
                                                      res, image_results_path, visualizeNames, silent=displayOnlyAP)
                                # TODO : Build automatically classes_hierarchy

                            classes_hierarchy = {
                                3: {"contains": [4, 5], "keep_if_no_child": False},
                                8: {"contains": [9, 10], "keep_if_no_child": True}
                            }
                            step = "filtering orphan masks (pass 2)"
                            progressBarPrefix = " - Removing orphan masks " if not displayOnlyAP else None
                            res = pp.filter_orphan_masks(res, bb_threshold=filter_bb_threshold,
                                                         mask_threshold=filter_mask_threshold,
                                                         classes_hierarchy=classes_hierarchy,
                                                         displayProgress=progressBarPrefix, config=self.__CONFIG,
                                                         verbose=0)

                        if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                            if savePreFusionImage:
                                debugIterator += 1
                                self.save_debug_image("pre class fusion", debugIterator, fullImage, imageInfo, res,
                                                      image_results_path, visualizeNames, silent=displayOnlyAP)
                            step = "fusing classes"
                            progressBarPrefix = " - Fusing overlapping equivalent masks " if not displayOnlyAP else None
                            classes_compatibility = [[4, 5]]  # Nsg partiel + nsg complet
                            res = pp.fuse_class(res, bb_threshold=fusion_bb_threshold,
                                                mask_threshold=fusion_mask_threshold,
                                                classes_compatibility=classes_compatibility, config=self.__CONFIG,
                                                displayProgress=progressBarPrefix, verbose=0)
                            if savePreFilterImage:
                                debugIterator += 1
                                self.save_debug_image("pre small masks removal", debugIterator, fullImage, imageInfo,
                                                      res, image_results_path, visualizeNames, silent=displayOnlyAP)
                            step = "removing small masks"
                            progressBarPrefix = " - Removing small masks " if not displayOnlyAP else None
                            res = pp.filter_small_masks(res, min_size=minMaskArea, config=self.__CONFIG,
                                                        displayProgress=progressBarPrefix, verbose=0)

                    if imageInfo["HAS_ANNOTATION"]:
                        if not displayOnlyAP:
                            print(" - Computing Average Precision and Confusion Matrix")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # TODO : Build automatically classes_hierarchy
                            classes_hierarchy = None
                            if not self.__CORTEX_MODE and len(self.__CUSTOM_CLASS_NAMES) == 10:
                                classes_hierarchy = {3: [4, 5], 8: [9, 10], 9: [10]}
                            step = "computing confusion matrix"
                            AP, _, _, _, confusion_matrix = utils.compute_ap(gt_boxes=gt_bbox, gt_class_ids=gt_class_id,
                                                                             gt_masks=gt_mask, pred_boxes=res["rois"],
                                                                             pred_class_ids=res["class_ids"],
                                                                             pred_scores=res["scores"],
                                                                             pred_masks=res['masks'],
                                                                             nb_class=self.__NB_CLASS,
                                                                             score_threshold=0.3,
                                                                             iou_threshold=0.5,
                                                                             confusion_iou_threshold=0.3,
                                                                             classes_hierarchy=classes_hierarchy,
                                                                             confusion_background_class=True,
                                                                             confusion_only_best_match=False)

                            print(f"{'' if displayOnlyAP else '   '} - Average Precision is about {AP:06.2%}")
                            self.__CONFUSION_MATRIX = np.add(self.__CONFUSION_MATRIX, confusion_matrix)
                            self.__APs.append(AP)
                            if save_results:
                                step = "saving confusion matrix"
                                name = f"{imageInfo['NAME']} Confusion Matrix"
                                confusionMatrixFileName = os.path.join(image_results_path, name.replace(' ', '_'))
                                name2 = f"{imageInfo['NAME']} Confusion Matrix (Normalized)"
                                confusionMatrixFileName2 = os.path.join(image_results_path, name2.replace('(', '')
                                                                        .replace(')', '')
                                                                        .replace(' ', '_'))
                                visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                                   title=name,
                                                                   cmap=cmap, show=False,
                                                                   fileName=confusionMatrixFileName)
                                visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                                   title=name2,
                                                                   cmap=cmap, show=False, normalize=True,
                                                                   fileName=confusionMatrixFileName2)
                                del dataset_val, gt_mask, gt_bbox, gt_class_id

                    if not self.__CORTEX_MODE:
                        step = "computing statistics"
                        print(" - Computing statistics on predictions")
                        stats = pp.getCountAndArea(res, classesInfo=self.__CLASSES_INFO,
                                                   selectedClasses=self.__CUSTOM_CLASS_NAMES, config=self.__CONFIG)
                        print(f"    - cortex : area = {imageInfo['CORTEX_AREA']}")
                        for className in self.__CUSTOM_CLASS_NAMES:
                            stat = stats[className]
                            print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")
                        if save_results:
                            with open(os.path.join(image_results_path, f"{imageInfo['NAME']}_stats.json"),
                                      "w") as saveFile:
                                stats["cortex"] = {"count": 1, "area": imageInfo["CORTEX_AREA"]}
                                try:
                                    json.dump(stats, saveFile, indent='\t')
                                except TypeError:
                                    print("    Failed to save statistics", flush=True)

                    if save_results:
                        if self.__CORTEX_MODE:
                            step = "cleaning full resolution image"
                            if not displayOnlyAP:
                                print(" - Cleaning full resolution image and saving statistics")
                            allCortices = None
                            # Gathering every cortex masks into one
                            for idxMask, classMask in enumerate(res['class_ids']):
                                if classMask == 1:
                                    if allCortices is None:  # First mask found
                                        allCortices = res['masks'][:, :, idxMask].copy() * 255
                                    else:  # Additional masks found
                                        allCortices = cv2.bitwise_or(allCortices, res['masks'][:, :, idxMask] * 255)

                            # To avoid cleaning an image without cortex
                            if allCortices is not None:
                                # Extracting the new Bbox
                                allCorticesROI = utils.extract_bboxes(allCortices)

                                fusion_dir = os.path.join(image_results_path, f"{imageInfo['NAME']}_fusion")
                                os.makedirs(fusion_dir, exist_ok=True)
                                allCorticesSmall = allCortices[allCorticesROI[0]:allCorticesROI[2],
                                                   allCorticesROI[1]:allCorticesROI[3]]
                                cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_cortex.jpg"),
                                            allCorticesSmall, CV2_IMWRITE_PARAM)
                                # Computing coordinates at full resolution
                                yRatio = imageInfo['HEIGHT'] / self.__CORTEX_SIZE[0]
                                xRatio = imageInfo['WIDTH'] / self.__CORTEX_SIZE[1]
                                allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
                                allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
                                allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
                                allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

                                # Resizing and adding the 2 missing channels of the cortices mask
                                allCortices = cv2.resize(np.uint8(allCortices),
                                                         (imageInfo['WIDTH'], imageInfo['HEIGHT']),
                                                         interpolation=cv2.INTER_CUBIC)
                                temp = np.repeat(allCortices[:, :, np.newaxis], 3, axis=2)

                                # Masking the image and saving it
                                imageInfo['FULL_RES_IMAGE'] = cv2.bitwise_and(
                                    imageInfo['FULL_RES_IMAGE'][allCorticesROI[0]: allCorticesROI[2],
                                    allCorticesROI[1]:allCorticesROI[3], :],
                                    temp[allCorticesROI[0]: allCorticesROI[2],
                                    allCorticesROI[1]:allCorticesROI[3], :])
                                cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_cleaned.jpg"),
                                            cv2.cvtColor(imageInfo['FULL_RES_IMAGE'], cv2.COLOR_RGB2BGR),
                                            CV2_IMWRITE_PARAM)

                                #########################################################
                                # Preparing to export all "fusion" divisions with stats #
                                #########################################################
                                fusion_info_file_path = os.path.join(image_results_path,
                                                                     f"{imageInfo['NAME']}_fusion_info.skinet")
                                fusionInfo = {"image": imageInfo["NAME"]}

                                # Computing ratio between full resolution image and the low one
                                height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                                smallHeight, smallWidth = allCorticesSmall.shape
                                xRatio = width / smallWidth
                                yRatio = height / smallHeight

                                # Computing divisions coordinates for full and low resolution images
                                divisionSize = dD.getMaxSizeForDivAmount(nbMaxDivPerAxis, self.__DIVISION_SIZE,
                                                                         self.__MIN_OVERLAP_PART_MAIN)
                                xStarts = dD.computeStartsOfInterval(width, intervalLength=divisionSize,
                                                                     min_overlap_part=0)
                                yStarts = dD.computeStartsOfInterval(height, intervalLength=divisionSize,
                                                                     min_overlap_part=0)

                                xStartsEquivalent = [round(x / xRatio) for x in xStarts]
                                yStartsEquivalent = [round(y / yRatio) for y in yStarts]
                                xDivSide = round(divisionSize / xRatio)
                                yDivSide = round(divisionSize / yRatio)

                                # Preparing informations to write into the .skinet file
                                fusionInfo["division_size"] = divisionSize
                                fusionInfo["min_overlap_part"] = 0
                                fusionInfo["xStarts"] = xStarts
                                fusionInfo["yStarts"] = yStarts
                                fusionInfo["max_div_per_axis"] = nbMaxDivPerAxis
                                fusionInfo["cortex_area"] = dD.getBWCount(allCortices)[1]
                                fusionInfo["divisions"] = {}

                                with open(os.path.join(image_results_path, f"{imageInfo['NAME']}_stats.json"),
                                          "w") as saveFile:
                                    stats = {"cortex": {"count": 1, "area": fusionInfo["cortex_area"]}}
                                    try:
                                        json.dump(stats, saveFile, indent='\t')
                                    except TypeError:
                                        print("    Failed to save statistics", flush=True)

                                step = "saving divisions of cleaned image"
                                # Extracting and saving all divisions
                                for divID in range(dD.getDivisionsCount(xStarts, yStarts)):
                                    cortexDiv = dD.getImageDivision(allCorticesSmall, xStartsEquivalent,
                                                                    yStartsEquivalent,
                                                                    divID, divisionSize=(xDivSide, yDivSide))
                                    black, white = dD.getBWCount(cortexDiv.astype(np.uint8))
                                    partOfDiv = white / (white + black)
                                    fusionInfo["divisions"][divID] = {"cortex_area": white,
                                                                      "cortex_representative_part": partOfDiv,
                                                                      "used": partOfDiv > fusionDivThreshold}
                                    if partOfDiv > fusionDivThreshold:
                                        x, xEnd, y, yEnd = dD.getDivisionByID(xStarts, yStarts, divID, divisionSize)
                                        fusionInfo["divisions"][divID]["coordinates"] = {"x1": x, "x2": xEnd, "y1": y,
                                                                                         "y2": yEnd}
                                        imageDivision = dD.getImageDivision(imageInfo['FULL_RES_IMAGE'], xStarts,
                                                                            yStarts,
                                                                            divID, divisionSize)
                                        cv2.imwrite(os.path.join(fusion_dir, f"{imageInfo['NAME']}_{divID}.jpg"),
                                                    cv2.cvtColor(imageDivision, cv2.COLOR_RGB2BGR), CV2_IMWRITE_PARAM)

                                # Writing informations into the .skinet file
                                with open(fusion_info_file_path, 'w') as fusionInfoFile:
                                    try:
                                        json.dump(fusionInfo, fusionInfoFile, indent="\t")
                                    except TypeError:
                                        print("    Failed to save fusion info file", file=sys.stderr, flush=True)

                        if not displayOnlyAP:
                            print(" - Applying masks on image")
                        step = "saving predicted image"
                        fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Predicted")
                        # No need of reloading or passing copy of image as it is the final drawing
                        _ = visualize.display_instances(fullImage, res['rois'], res['masks'], res['class_ids'],
                                                        visualizeNames, res['scores'], colorPerClass=True,
                                                        fileName=fileName, onlyImage=True, silent=True, figsize=(
                                                         (1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                         (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100
                                                        ), image_format=imageInfo['IMAGE_FORMAT'], config=self.__CONFIG)
                        # Annotations Extraction
                        step = "saving annotations"
                        if not displayOnlyAP:
                            print(" - Saving predicted annotations files")
                        for adapter in AnnotationAdapter.ANNOTATION_ADAPTERS:
                            try:
                                pp.export_annotations(imageInfo, res, self.__CLASSES_INFO,
                                                      adapter, save_path=image_results_path,
                                                      config=self.__CONFIG, verbose=0 if displayOnlyAP else 1)
                            except Exception:
                                print(f"Failed to export using {adapter.__qualname__} adapter")
                    final_time = round(time() - start_time)
                    print(f" Done in {formatTime(final_time)}\n")
                    if not imageInfo['HAS_ANNOTATION']:
                        AP = -1
                    step = "finalizing"
                    if save_results:
                        with open(logsPath, 'a') as results_log:
                            apText = f"{AP:4.3f}".replace(".", ",")
                            results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText}%;\n")
                    del res, imageInfo, fullImage
                    plt.clf()
                    plt.close('all')
                    gc.collect()
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                failedImages.append(os.path.basename(IMAGE_PATH))
                print(f"/!\\ Failed {IMAGE_PATH} at \"{step}\"\n")
                if save_results and step not in ["image preparation", "finalizing"]:
                    apText = ""
                    if imageInfo['HAS_ANNOTATION'] and step in []:
                        apText = f"{AP:4.3f}".replace(".", ",")
                    final_time = round(time() - start_time)
                    with open(logsPath, 'a') as results_log:
                        results_log.write(f"{imageInfo['NAME']}; {final_time}; {apText};FAILED ({step});\n")
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
            name = "Final Confusion Matrix"
            name2 = "Final Confusion Matrix (Normalized)"
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name,
                                               cmap=cmap, show=False,
                                               fileName=(os.path.join(results_path, name.replace(' ', '_'))
                                                         if save_results else None))
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name2,
                                               cmap=cmap, show=False, normalize=True,
                                               fileName=(os.path.join(results_path, name2.replace('(', '')
                                                                      .replace(')', '')
                                                                      .replace(' ', '_'))
                                                         if save_results else None))
            plt.close('all')
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        print(f"All inferences done in {formatTime(total_time)}")
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = f"{mAP:4.3f}".replace(".", ",")
                results_log.write(f"GLOBAL; {total_time}; {mapText}%;\n")

    def save_debug_image(self, step, debugIterator, fullImage, imageInfo, res, image_results_path, names, silent=True):
        if not silent:
            print(f" - Saving {step} image")
        step = step.replace(' ', '_').replace('(', '').replace(')', '')
        fileName = os.path.join(image_results_path, f"{imageInfo['NAME']}_Inference_debug_{debugIterator:02d}_{step}")
        visualize.display_instances(fullImage if self.__LOW_MEMORY else fullImage.copy(), res['rois'], res['masks'],
                                    res['class_ids'], names, res['scores'], colorPerClass=True, fileName=fileName,
                                    onlyImage=False, silent=True, figsize=(
                (1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100
            ), image_format=imageInfo['IMAGE_FORMAT'], config=self.__CONFIG)
        if self.__LOW_MEMORY:
            del fullImage
            gc.collect()
            fullImage = cv2.cvtColor(cv2.imread(imageInfo['PATH']), cv2.COLOR_BGR2RGB)
