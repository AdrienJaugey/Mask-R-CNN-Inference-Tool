import json
import os
import shutil
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import time
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.io import imread, imsave
    from datasetTools import datasetDivider as div, AnnotationAdapter
    from datasetTools import datasetWrapper as wr
    from datasetTools.ASAPAdapter import ASAPAdapter
    from datasetTools.LabelMeAdapter import LabelMeAdapter
    from datetime import datetime

    from mrcnn.config import Config
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize


def get_ax(rows=1, cols=1, size=8):
    return plt.subplots(rows, cols, figsize=(size * cols, size * rows), frameon=False)


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
                if os.path.exists(os.path.join(dirPath, name + '.png')):
                    image.append(name + '.png')
                elif os.path.exists(os.path.join(dirPath, name + '.jpg')):
                    image.append(name + '.jpg')
                else:
                    image.append(file)
        elif extension == 'png' or extension == "jpg":
            image.append(file)
        elif extension == 'skinet':
            with open(os.path.join(dirPath, file), 'r') as skinetFile:
                fusionInfo = json.load(skinetFile)
                fusionDir = fusionInfo['image'] + "_fusion"
                if fusionDir in files:
                    divExists = False
                    divPath = os.path.join(fusionDir, fusionInfo['image'] + '_{}.jpg')
                    for divID in range(len(fusionInfo["divisions"])):
                        if fusionInfo["divisions"][str(divID)]["used"] and os.path.exists(os.path.join(dirPath, divPath.format(divID))):
                            image.append(divPath.format(divID))
                            divExists = True
                        elif fusionInfo["divisions"][str(divID)]["used"]:
                            print("Div n°{} of {} missing".format(divID))
                    if divExists:
                        image.append(file)

    for i in range(len(image)):
        image[i] = os.path.join(dirPath, image[i])
    return image


def stop():
    modellib.terminate_session()


class NephrologyInferenceModel:

    def __init__(self, classesInfo, modelPath, divisionSize=1024, min_overlap_part_main=0.33, min_overlap_part_cortex=0.5, cortex_size=None):
        print("Initialisation")
        self.__CLASSES_INFO = classesInfo
        self.__CORTEX_MODE = not classesInfo[0]["ignore"]
        self.__MODEL_PATH = modelPath
        self.__DIVISION_SIZE = divisionSize
        self.__MIN_OVERLAP_PART_MAIN = min_overlap_part_main
        self.__MIN_OVERLAP_PART_CORTEX = min_overlap_part_cortex
        self.__MIN_OVERLAP_PART = min_overlap_part_cortex if self.__CORTEX_MODE else min_overlap_part_main
        self.__CORTEX_SIZE = None if not self.__CORTEX_MODE else (1024, 1024) if cortex_size is None else cortex_size
        self.__CELLS_CLASS_NAMES = []
        for classInfo in classesInfo:
            if not classInfo["ignore"]:
                self.__CELLS_CLASS_NAMES.append(classInfo["name"])
        self.__NB_CLASS = len(self.__CELLS_CLASS_NAMES)
        # Root directory of the project
        self.__ROOT_DIR = os.getcwd()

        # Directory to save logs and trained model
        self.__MODEL_DIR = os.path.join(self.__ROOT_DIR, "logs")

        # Configurations
        nbClass = self.__NB_CLASS
        divSize = 1024 if self.__DIVISION_SIZE == "noDiv" else self.__DIVISION_SIZE
        self.__CONFUSION_MATRIX = np.zeros((self.__NB_CLASS + 1, self.__NB_CLASS + 1), dtype=np.int32)
        self.__APs = []

        class CellsConfig(Config):
            NAME = "cells"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + nbClass
            IMAGE_MIN_DIM = divSize
            IMAGE_MAX_DIM = divSize
            RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)
            TRAIN_ROIS_PER_IMAGE = 800
            STEPS_PER_EPOCH = 400
            VALIDATION_STEPS = 50

        config = CellsConfig()

        class InferenceConfig(CellsConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        self.__MODEL = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=self.__MODEL_DIR)

        # Load trained weights (fill in path to trained weights here)
        assert self.__MODEL_PATH != "", "Provide path to trained weights"
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
            image_file_name = os.path.basename(imagePath)
            dirPath = os.path.dirname(imagePath)
            image_name = image_file_name.split('.')[0]
            image_extension = image_file_name.split('.')[-1]

            if self.__CORTEX_MODE:
                fullResImage = cv2.imread(imagePath)
                height, width, _ = fullResImage.shape
                fullImage = cv2.resize(fullResImage, self.__CORTEX_SIZE)
                fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2RGB)
            else:
                fullImage = cv2.imread(imagePath)
                fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2RGB)
            if image_extension not in ["png", "jpg"]:
                # print('Converting to png')
                tempPath = dirPath + image_name + '.png'
                image_extension = "png"
                imsave(tempPath, fullImage)
                imagePath = tempPath

            # Creating the result dir if given and copying the base image in it
            if results_path is not None:
                image_results_path = results_path + image_name + "/"
                os.makedirs(image_results_path, exist_ok=True)
                imsave(os.path.join(image_results_path, image_name + ".png"), fullImage)
            else:
                image_results_path = None

            if not self.__CORTEX_MODE:
                fullImage = cv2.imread(imagePath)
                fullImage = cv2.cvtColor(fullImage, cv2.COLOR_BGR2RGB)
            image = fullImage.copy()
            if not self.__CORTEX_MODE:
                height, width, _ = fullImage.shape
            cortexArea = height * width
            xStarts = [0] if self.__DIVISION_SIZE == "noDiv" else div.computeStartsOfInterval(
                maxVal=self.__CORTEX_SIZE[0] if self.__CORTEX_MODE else width, intervalLength=self.__DIVISION_SIZE,
                min_overlap_part=self.__MIN_OVERLAP_PART
            )
            yStarts = [0] if self.__DIVISION_SIZE == "noDiv" else div.computeStartsOfInterval(
                maxVal=self.__CORTEX_SIZE[1] if self.__CORTEX_MODE else height, intervalLength=self.__DIVISION_SIZE,
                min_overlap_part=self.__MIN_OVERLAP_PART
            )

            nbDiv = div.getDivisionsCount(xStarts, yStarts)

            annotationExists = False
            has_annotation = False
            for ext in AnnotationAdapter.ANNOTATION_FORMAT:
                annotationExists = annotationExists or os.path.exists(dirPath + image_name + '.' + ext)
            if annotationExists:
                if not silent:
                    print(" - Annotation file found: creating dataset files and cleaning image if possible")
                wr.createMasksOfImage(dirPath, image_name, 'data', classesInfo=self.__CLASSES_INFO,
                                      imageFormat=image_extension, resize=self.__CORTEX_SIZE)
                if not self.__CORTEX_MODE:
                    wr.fuseCortices('data', image_name, deleteBaseMasks=True)
                    wr.cleanImage('data', image_name, onlyMasks=False)
                maskDirs = os.listdir(os.path.join('data', image_name))
                if not self.__CORTEX_MODE:
                    if "cortex" in maskDirs:
                        cortexDirPath = os.path.join('data', image_name, 'cortex')
                        cortexImageFilePath = os.listdir(cortexDirPath)[0]
                        cortex = cv2.imread(os.path.join(cortexDirPath, cortexImageFilePath), cv2.IMREAD_UNCHANGED)
                        cortexArea = div.getBWCount(cortex)[1]
                    # If full_images directory exists it means than image has been cleaned so we have to get it another
                    # time
                    if 'full_images' in maskDirs:
                        imagesDirPath = os.path.join('data', image_name, 'images')
                        imageFilePath = os.listdir(imagesDirPath)[0]
                        image = cv2.imread(os.path.join(imagesDirPath, imageFilePath))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # We want to know if image has annotation, if we don't want to detect cortex and this mask exist
                # as we need it to clean the image, we remove it from the mask list before checking if a class
                # we want to predict has an annotated mask
                if "cortex" not in self.__CELLS_CLASS_NAMES and "cortex" in maskDirs:
                    maskDirs.remove("cortex")
                for classToPredict in self.__CELLS_CLASS_NAMES:
                    if classToPredict in maskDirs:
                        has_annotation = True
                        break
                if has_annotation and not silent:
                    print("    - AP and confusion matrix will be computed")

            imageInfo = {
                "PATH": imagePath,
                "DIR_PATH": dirPath,
                "FILE_NAME": image_file_name,
                "IMAGE_FORMAT": image_extension,
                "NAME": image_name,
                "HEIGHT": height,
                "WIDTH": width,
                "NB_DIV": nbDiv,
                "X_STARTS": xStarts,
                "Y_STARTS": yStarts,
                "HAS_ANNOTATION": has_annotation,
                "CORTEX_AREA": cortexArea
            }
            if self.__CORTEX_MODE:
                imageInfo['FULL_RES_IMAGE'] = fullResImage
        return image, fullImage, imageInfo, image_results_path

    def init_results_dir(self, results_path):
        if results_path is None or results_path in ['.', './', "/"]:
            lastDir = "results"
            remainingPath = ""
        else:
            if results_path[-1] == '/':
                index = -2
            else:
                index = -1
            lastDir = results_path.split('/')[index].replace('/', '')
            remainingPath = results_path[0:results_path.index(lastDir)]
        results_path = remainingPath + "{}_{}/".format(lastDir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(results_path)
        print("Results will be saved to {}".format(results_path))
        logsPath = os.path.join(results_path, 'inference_data.csv')
        with open(logsPath, 'w') as results_log:
            results_log.write("Image; Duration (s); Precision;\n")
        return results_path, logsPath

    def inference(self, images: list, results_path=None, save_results=True,
                  fusion_bb_threshold=0., fusion_mask_threshold=0.1,
                  filter_bb_threshold=0.5, filter_mask_threshold=0.9,
                  priority_table=None, nbMaxDivPerAxis=3, fusionDivThreshold=0.1,
                  displayOnlyAP=False):

        assert len(images) > 0, "images list cannot be empty"

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
        for i, IMAGE_PATH in enumerate(images):
            # Last step of full image inference
            if '_fusion_info.skinet' in IMAGE_PATH:
                if self.__CORTEX_MODE:
                    continue
                with open(IMAGE_PATH, 'r') as skinetFile:
                    fusionInfo = json.load(skinetFile)
                print("Finalising {} image ({}/{} | {:4.2f}%)".format(fusionInfo["image"], i + 1, len(images),
                                                                      (i + 1) / len(images) * 100))
                fusionDir = fusionInfo["image"] + '_fusion'
                fusionDirPath = os.path.join(os.path.dirname(IMAGE_PATH), fusionDir)
                cleaned_image_path = os.path.join(fusionDirPath, fusionInfo["image"] + "_cleaned.jpg")
                cleanedImage = cv2.imread(cleaned_image_path)
                image_results_path = os.path.join(results_path, fusionInfo['image'])
                os.makedirs(image_results_path, exist_ok=True)

                # Pasting each divisions and merging all the stats
                globalStats = None
                for imageFolder in os.listdir(results_path):
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
                            image = cv2.imread(imagePath)
                            coo = fusionInfo["divisions"][divID]["coordinates"]
                            cleanedImage[coo["y1"]:coo["y2"], coo["x1"]:coo["x2"], :] = image
                            shutil.move(os.path.join(results_path, imageFolder),
                                        os.path.join(image_results_path, "divisions", imageFolder))
                cv2.imwrite(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction.jpg"), cleanedImage)
                with open(os.path.join(image_results_path, fusionInfo['image'] + "_full_prediction_stats.json"),
                          'w') as globalStatsFile:
                    globalStats["cortex"] = {"count": 1, "area": fusionInfo["cortex_area"]}
                    json.dump(globalStats, globalStatsFile, indent="\t")
                print("Done\n")
            else:
                start_time = time()
                print("Using {} image file ({}/{} | {:4.2f}%)".format(IMAGE_PATH, i + 1, len(images),
                                                                      (i + 1) / len(images) * 100))
                image, fullImage, imageInfo, image_results_path = self.prepare_image(IMAGE_PATH, results_path,
                                                                                     silent=displayOnlyAP)
                if imageInfo["HAS_ANNOTATION"]:
                    class LightCellsDataset(utils.Dataset):

                        def __init__(self, cells_class_names, cortex_mode, cortex_size):
                            super().__init__()
                            self.__CELLS_CLASS_NAMES = cells_class_names.copy()
                            self.__CORTEX_MODE = cortex_mode
                            self.__CORTEX_SIZE = cortex_size

                        def get_class_names(self):
                            return self.__CELLS_CLASS_NAMES.copy()

                        def load_cells(self):
                            # Add classes
                            for class_id, class_name in enumerate(self.__CELLS_CLASS_NAMES):
                                self.add_class("cells", class_id + 1, class_name)

                            img_path = 'data/' + imageInfo["NAME"] + '/images/'
                            self.add_image("cells", image_id=imageInfo["NAME"], path=img_path)

                        def load_image(self, image_id):

                            info = self.image_info[image_id]
                            info = info.get("id")

                            img = imread('data/' + info + '/images/' + info + '.png')[:, :, :3]

                            return img

                        def image_reference(self, image_id):
                            """Return the cells data of the image."""
                            info = self.image_info[image_id]
                            if info["source"] == "cells":
                                return info["cells"]
                            else:
                                super(self.__class__).image_reference(self, image_id)

                        def load_mask(self, image_id):
                            """Generate instance masks for cells of the given image ID.
                            """
                            info = self.image_info[image_id]
                            info = info.get("id")

                            path = 'data/' + info

                            # Counting masks for current image
                            number_of_masks = 0
                            for masks_dir in os.listdir(path):
                                # For each directory excepting /images
                                if masks_dir not in self.__CELLS_CLASS_NAMES:
                                    continue
                                temp_DIR = path + '/' + masks_dir
                                # Adding length of directory https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
                                number_of_masks += len(
                                    [name for name in os.listdir(temp_DIR) if
                                     os.path.isfile(os.path.join(temp_DIR, name))])

                            mask = np.zeros([self.__CORTEX_SIZE[0] if self.__CORTEX_MODE else imageInfo["HEIGHT"],
                                             self.__CORTEX_SIZE[1] if self.__CORTEX_MODE else imageInfo["WIDTH"],
                                             number_of_masks], dtype=np.uint8)
                            iterator = 0
                            class_ids = np.zeros((number_of_masks,), dtype=int)
                            for masks_dir in os.listdir(path):
                                if masks_dir not in self.__CELLS_CLASS_NAMES:
                                    continue
                                temp_class_id = self.__CELLS_CLASS_NAMES.index(masks_dir) + 1
                                for mask_file in next(os.walk(path + '/' + masks_dir + '/'))[2]:
                                    mask_ = imread(path + '/' + masks_dir + '/' + mask_file)
                                    mask[:, :, iterator] = mask_
                                    class_ids[iterator] = temp_class_id
                                    iterator += 1
                            # Handle occlusions
                            occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
                            for i in range(number_of_masks - 2, -1, -1):
                                mask[:, :, i] = mask[:, :, i] * occlusion
                                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
                            return mask, class_ids.astype(np.int32)

                    dataset_val = LightCellsDataset(self.__CELLS_CLASS_NAMES, self.__CORTEX_MODE, self.__CORTEX_SIZE)
                    dataset_val.load_cells()
                    dataset_val.prepare()

                    image_id = dataset_val.image_ids[0]
                    gt_mask, gt_class_id = dataset_val.load_mask(image_id)
                    gt_bbox = utils.extract_bboxes(gt_mask)
                    if save_results:
                        if not displayOnlyAP:
                            print(" - Applying annotations on file to get expected results")
                        fileName = image_results_path + "{}_Expected".format(imageInfo["NAME"])
                        visualize.display_instances(fullImage, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names,
                                                    colorPerClass=True,
                                                    figsize=((1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                             (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100),
                                                    title="{} Expected".format(imageInfo["NAME"]),
                                                    fileName=fileName, silent=True)

                # Getting predictions for each division

                res = []
                if not displayOnlyAP:
                    print(" - Starting Inference")
                for divId in range(imageInfo["NB_DIV"]):
                    division = div.getImageDivision(image, imageInfo["X_STARTS"], imageInfo["Y_STARTS"], divId)
                    if not displayOnlyAP:
                        print('    - Inference {}/{}'.format(divId + 1, imageInfo["NB_DIV"]))
                    results = self.__MODEL.detect([division])
                    res.append(results[0])

                # Post-processing of the predictions
                # if not self.__CORTEX_MODE:
                if not displayOnlyAP:
                    print(" - Fusing results of all divisions")
                res = utils.fuse_results(res, image, division_size=self.__DIVISION_SIZE,
                                         min_overlap_part=self.__MIN_OVERLAP_PART)
                # else:
                #     res = res[0]
                if not displayOnlyAP:
                    print(" - Fusing overlapping masks")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = utils.fuse_masks(res, bb_threshold=fusion_bb_threshold, mask_threshold=fusion_mask_threshold,
                                           verbose=1)

                    if not displayOnlyAP:
                        print(" - Removing non-sense masks")
                    res = utils.filter_fused_masks(res,
                                                   bb_threshold=filter_bb_threshold,
                                                   mask_threshold=filter_mask_threshold,
                                                   priority_table=priority_table)

                if imageInfo["HAS_ANNOTATION"]:
                    if not displayOnlyAP:
                        print(" - Computing Average Precision and Confusion Matrix")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        AP, _, _, _, confusion_matrix = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                                         res["rois"], res["class_ids"],
                                                                         res["scores"], res['masks'],
                                                                         nb_class=self.__NB_CLASS,
                                                                         confusion_iou_threshold=0.1)

                        print(
                            "{} - Average Precision is about {:5.2f}%".format("" if displayOnlyAP else "   ", AP * 100))
                        self.__CONFUSION_MATRIX = np.add(self.__CONFUSION_MATRIX, confusion_matrix)
                        self.__APs.append(AP)
                        if save_results:
                            name = "{} Confusion Matrix".format(imageInfo["NAME"])
                            name2 = "{} Normalized Confusion Matrix".format(imageInfo["NAME"])
                            visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                               title=name,
                                                               cmap=cmap, show=False,
                                                               fileName=image_results_path + name.replace(' ', '_'))
                            visualize.display_confusion_matrix(confusion_matrix, dataset_val.get_class_names(),
                                                               title=name2,
                                                               cmap=cmap, show=False, normalize=True,
                                                               fileName=image_results_path + name2.replace(' ', '_'))

                if not self.__CORTEX_MODE:
                    print(" - Computing statistics on predictions")
                    stats = utils.getCountAndArea(res, classesInfo=self.__CLASSES_INFO,
                                                  selectedClasses=self.__CELLS_CLASS_NAMES)
                    print("    - cortex : area = {}".format(imageInfo["CORTEX_AREA"]))
                    for className in self.__CELLS_CLASS_NAMES:
                        print("    - {} : count = {}, area = {} px".format(className, stats[className]["count"],
                                                                           stats[className]["area"]))
                    if save_results:
                        with open(image_results_path + imageInfo["NAME"] + "_stats.json", "w") as saveFile:
                            stats["cortex"] = {"count": 1, "area": imageInfo["CORTEX_AREA"]}
                            json.dump(stats, saveFile, indent='\t')

                if not displayOnlyAP:
                    print(" - Applying masks on image")

                if save_results:
                    if self.__CORTEX_MODE:
                        if not displayOnlyAP:
                            print(" - Cleaning full resolution image")
                        allCortices = None
                        allMedulla = None
                        # Gathering every cortex masks into one
                        for idxMask, classMask in enumerate(res['class_ids']):
                            if classMask == 1:
                                if allCortices is None:  # First mask found
                                    allCortices = res['masks'][:, :, idxMask].copy() * 255
                                else:  # Additional masks found
                                    allCortices = cv2.bitwise_or(allCortices, res['masks'][:, :, idxMask] * 255)
                            elif classMask == 2:
                                if allMedulla is None:
                                    allMedulla = res['masks'][:, :, idxMask].copy() * 255
                                else:
                                    allMedulla = cv2.bitwise_or(allMedulla, res['masks'][:, :, idxMask] * 255)

                        # To avoid cleaning an image without cortex
                        if allCortices is not None:
                            # If Medulla mask found, we clean the cortex with those
                            if allMedulla is not None:
                                allMedulla = cv2.bitwise_not(allMedulla.astype(np.uint8))
                                allCortices = cv2.bitwise_and(allCortices.astype(np.uint8), allMedulla)

                            # Extracting the new Bbox
                            tempMask = np.expand_dims(allCortices, axis=2)
                            allCorticesROI = utils.extract_bboxes(tempMask)[0, :]

                            fusion_dir = os.path.join(image_results_path, '{}_fusion'.format(imageInfo['NAME']))
                            os.makedirs(fusion_dir, exist_ok=True)
                            allCorticesSmall = allCortices[allCorticesROI[0]:allCorticesROI[2],
                                                           allCorticesROI[1]:allCorticesROI[3]]
                            cv2.imwrite(os.path.join(fusion_dir, "{}_cortex.jpg".format(imageInfo["NAME"])),
                                        allCorticesSmall)
                            # Computing coordinates at full resolution
                            yRatio = imageInfo['HEIGHT'] / self.__CORTEX_SIZE[0]
                            xRatio = imageInfo['WIDTH'] / self.__CORTEX_SIZE[1]
                            allCorticesROI[0] = int(allCorticesROI[0] * yRatio)
                            allCorticesROI[1] = int(allCorticesROI[1] * xRatio)
                            allCorticesROI[2] = int(allCorticesROI[2] * yRatio)
                            allCorticesROI[3] = int(allCorticesROI[3] * xRatio)

                            # Resizing and adding the 2 missing channels of the cortices mask
                            allCortices = cv2.resize(np.uint8(allCortices), (imageInfo['WIDTH'], imageInfo['HEIGHT']),
                                                     interpolation=cv2.INTER_CUBIC)
                            temp = np.zeros(imageInfo['FULL_RES_IMAGE'].shape, dtype=np.uint8)
                            temp[:, :, 0] = allCortices
                            temp[:, :, 1] = allCortices
                            temp[:, :, 2] = allCortices

                            # Masking the image and saving it
                            imageInfo['FULL_RES_IMAGE'] = cv2.bitwise_and(
                                imageInfo['FULL_RES_IMAGE'][allCorticesROI[0]: allCorticesROI[2],
                                                            allCorticesROI[1]:allCorticesROI[3], :],
                                temp[allCorticesROI[0]: allCorticesROI[2],
                                     allCorticesROI[1]:allCorticesROI[3], :])
                            cv2.imwrite(os.path.join(fusion_dir, "{}_cleaned.jpg".format(imageInfo["NAME"])),
                                        imageInfo['FULL_RES_IMAGE'])

                            #########################################################
                            # Preparing to export all "fusion" divisions with stats #
                            #########################################################
                            fusion_info_file_path = os.path.join(image_results_path, "{}_fusion_info.skinet".format(imageInfo["NAME"]))
                            fusionInfo = {"image": imageInfo["NAME"]}

                            # Computing ratio between full resolution image and the low one
                            height, width, _ = imageInfo['FULL_RES_IMAGE'].shape
                            smallHeight, smallWidth = allCorticesSmall.shape
                            xRatio = width / smallWidth
                            yRatio = height / smallHeight

                            # Computing divisions coordinates for full and low resolution images
                            divisionSize = div.getMaxSizeForDivAmount(nbMaxDivPerAxis, self.__DIVISION_SIZE, self.__MIN_OVERLAP_PART_MAIN)
                            xStarts = div.computeStartsOfInterval(width, intervalLength=divisionSize, min_overlap_part=0)
                            yStarts = div.computeStartsOfInterval(height, intervalLength=divisionSize, min_overlap_part=0)

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
                            fusionInfo["cortex_area"] = div.getBWCount(allCortices)[1]
                            fusionInfo["divisions"] = {}

                            # Extracting and saving all divisions
                            for divID in range(div.getDivisionsCount(xStarts, yStarts)):
                                cortexDiv = div.getImageDivision(allCorticesSmall, xStartsEquivalent, yStartsEquivalent,
                                                                 divID, divisionSize=(xDivSide, yDivSide))
                                black, white = div.getBWCount(cortexDiv.astype(np.uint8))
                                partOfDiv = white / (white + black)
                                fusionInfo["divisions"][divID] = {"cortex_area": white,
                                                                  "cortex_representative_part": partOfDiv,
                                                                  "used": partOfDiv > fusionDivThreshold}
                                if partOfDiv > fusionDivThreshold:
                                    x, xEnd, y, yEnd = div.getDivisionByID(xStarts, yStarts, divID, divisionSize)
                                    fusionInfo["divisions"][divID]["coordinates"] = {"x1": x, "x2": xEnd, "y1": y,
                                                                                     "y2": yEnd}
                                    imageDivision = div.getImageDivision(imageInfo['FULL_RES_IMAGE'], xStarts, yStarts, divID, divisionSize)
                                    cv2.imwrite(os.path.join(fusion_dir, "{}_{}.jpg".format(imageInfo["NAME"], divID)), imageDivision)

                            # Writing informations into the .skinet file
                            with open(fusion_info_file_path, 'w') as fusionInfoFile:
                                json.dump(fusionInfo, fusionInfoFile, indent="\t")

                    fileName = image_results_path + "{}_predicted".format(imageInfo["NAME"])
                    names = self.__CELLS_CLASS_NAMES.copy()
                    names.insert(0, 'background')
                    visualize.display_instances(fullImage, res['rois'], res['masks'], res['class_ids'], names,
                                                res['scores'],
                                                colorPerClass=True, fileName=fileName, onlyImage=True, silent=True,
                                                figsize=((1024 if self.__CORTEX_MODE else imageInfo["WIDTH"]) / 100,
                                                         (1024 if self.__CORTEX_MODE else imageInfo["HEIGHT"]) / 100))

                    # Annotations Extraction
                    if not displayOnlyAP:
                        print(" - Saving predicted annotations files")
                    utils.export_annotations(imageInfo["NAME"], res, self.__CLASSES_INFO,
                                             ASAPAdapter,
                                             save_path=image_results_path, verbose=0 if displayOnlyAP else 1)
                    utils.export_annotations(imageInfo["NAME"], res, self.__CLASSES_INFO,
                                             LabelMeAdapter,
                                             save_path=image_results_path, verbose=0 if displayOnlyAP else 1)
                final_time = round(time() - start_time)
                m = final_time // 60
                s = final_time % 60
                print(" Done in {:02d}m {:02d}s\n".format(m, s))
                if not imageInfo['HAS_ANNOTATION']:
                    AP = -1
                if save_results:
                    with open(logsPath, 'a') as results_log:
                        apText = "{:4.3f}".format(AP * 100).replace(".", ",")
                        results_log.write("{}; {}; {}%;\n".format(imageInfo["NAME"], final_time, apText))
                plt.close('all')

        if len(self.__APs) > 1:
            mAP = np.mean(self.__APs) * 100
            print("Mean Average Precision is about {:5.2f}%".format(mAP))
            name = "Final Confusion Matrix"
            name2 = "Final Normalized Confusion Matrix"
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name,
                                               cmap=cmap, show=False,
                                               fileName=(results_path + name.replace(' ',
                                                                                     '_')) if save_results else None)
            visualize.display_confusion_matrix(self.__CONFUSION_MATRIX, dataset_val.get_class_names(), title=name2,
                                               cmap=cmap, show=False, normalize=True,
                                               fileName=(results_path + name2.replace(' ',
                                                                                      '_')) if save_results else None)
        else:
            mAP = -1
        total_time = round(time() - total_start_time)
        h = total_time // 3600
        m = (total_time % 3600) // 60
        s = final_time % 60
        print("All inferences done in {:02d}h {:02d}m {:02d}s".format(h, m, s))
        if save_results:
            with open(logsPath, 'a') as results_log:
                mapText = "{:4.3f}".format(mAP).replace(".", ",")
                results_log.write("GLOBAL; {}; {}%;\n".format(total_time, mapText))
