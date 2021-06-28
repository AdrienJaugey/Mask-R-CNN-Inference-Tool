import os
import numpy as np
import cv2
from time import time

from common_utils import progressBar
from datasetTools.datasetDivider import CV2_IMWRITE_PARAM
from datasetTools.datasetWrapper import loadSameResImage, getBboxFromName
from mrcnn import utils
from mrcnn.Config import Config
from mrcnn.utils import in_roi, extract_bboxes


def correct_path(value):
    """
    Try to convert the given value as a normalized path
    :param value: value to test
    :return: value as normalized path
    :raise: os.path.normpath base exceptions if could not convert value
    """
    return os.path.normpath(value)


def existing_path(value):
    """
    Test if the given value is an existing path
    :param value: value to test
    :return: normalized path
    :raise ValueError if not an existing path
    :raise os.path.normpath base exceptions
    """
    path = correct_path(value)
    if os.path.exists(path):
        return path
    else:
        raise ValueError('This is not an existing path')


def listdir2(path: str):
    """
    Decorates os.listdir by returning list of filenames and filepath
    :param path: the folder path to list content
    :return: list of (filename, filepath)
    """
    path = os.path.normpath(path)
    return [(f, os.path.join(path, f)) for f in os.listdir(path)]


def center_mask(mask_bbox, image_shape, min_output_shape=1024, verbose=0):
    """
    Computes shifted bbox of a mask for it to be centered in the output image
    :param mask_bbox: the original mask bbox
    :param image_shape: the original image shape as int (assuming height = width) or (int, int)
    :param min_output_shape: the minimum shape of the image in which the mask will be centered
    :param verbose: 0 : No output, 1 : Errors, 2: Warnings, 3+: Info messages
    :return: the roi of the original image representing the output image, the mask bbox in the output image
    """
    if type(min_output_shape) is int:
        output_shape_ = (min_output_shape, min_output_shape)
    else:
        output_shape_ = tuple(min_output_shape[:2])

    # Computes mask height and width, also check for axis centering
    img_bbox = mask_bbox.copy()
    mask_shape = tuple(mask_bbox[2:] - mask_bbox[:2])
    anyAxisUnchanged = False
    for i in range(2):  # For both x and y axis
        if mask_shape[i] < output_shape_[i]:  # If mask size is greater than wanted output size on this axis
            # Computing offset and applying it to get corresponding image bbox
            offset = (output_shape_[i] - mask_shape[i]) // 2
            img_bbox[i] = mask_bbox[i] - offset
            img_bbox[i + 2] = mask_bbox[i + 2] + offset + (0 if mask_shape[i] % 2 == 0 else 1)

            # Shifting the image bbox if part is outside base image
            if img_bbox[i] < 0:
                img_bbox[i + 2] -= img_bbox[i]
                img_bbox[i] = 0
            if img_bbox[i + 2] >= image_shape[i]:
                img_bbox[i] -= (img_bbox[i + 2] - image_shape[i] + 1)
                img_bbox[i + 2] = image_shape[i] - 1
        else:
            anyAxisUnchanged = True

    if anyAxisUnchanged and verbose > 1:
        print(f"Mask shape of {mask_shape} does not fit into output shape of {output_shape_}.")
    return img_bbox


def getCenteredClassBboxes(datasetPath: str, imageName: str, classToCenter: str, image_size=1024,
                           imageFormat="jpg", allow_oversized=True, config: Config = None, verbose=0):
    """
    Computes and returns bboxes of all masks of the given image and class
    :param datasetPath: path to the dataset containing the image folder
    :param imageName: the image name
    :param classToCenter: the class to center and get the bbox from
    :param image_size: the minimal height and width of the bboxes
    :param imageFormat: the image format to use to get original image
    :param allow_oversized: if False, masks that does not fit image_size will be skipped
    :param config: if given, config file is used to know if mini_masks are used
    :param verbose: level of verbosity
    :return: (N, 4) ndarray of [y1, x1, y2, x2] matching bboxes
    """
    imagePath = os.path.join(datasetPath, imageName, 'images', f'{imageName}.{imageFormat}')
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    image_shape = image.shape[:2]
    classDirPath = os.path.join(datasetPath, imageName, classToCenter)
    maskList = os.listdir(classDirPath)
    classBboxes = np.zeros((len(maskList), 4), dtype=int)
    toDelete = []
    for idx, mask in enumerate(maskList):
        maskPath = os.path.join(classDirPath, mask)
        if config is not None and config.is_using_mini_mask():
            bbox = getBboxFromName(mask)
        else:
            maskImage = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            bbox = utils.extract_bboxes(maskImage)
        if not allow_oversized:
            h, w = bbox[2:] - bbox[:2]
            if h > image_size or w > image_size:
                if verbose > 1:
                    print(f"{mask} mask could not fit into {(image_size, image_size)} image")
                toDelete.append(idx)
        classBboxes[idx] = center_mask(bbox, image_shape, min_output_shape=image_size, verbose=verbose)
    classBboxes = np.delete(classBboxes, toDelete, axis=0)
    return classBboxes


def isolateClass(datasetPath: str, outputDatasetPath: str, classToIsolate: str, image_size=1024,
                 imageFormat="jpg", allow_oversized=True, verbose=0, silent=False):
    """
    Separate base image and masks based on a class by taking each of this class's masks and centering them on a smaller
    image of shape (image_size, image_size, 1 or 3)
    :param datasetPath: path to the input dataset
    :param outputDatasetPath: path to the output dataset
    :param classToIsolate: the class to center and to use to clean image and other masks
    :param image_size: the wanted output image shape (image_size, image_size)
    :param imageFormat: the image format to use such as jpg, png...
    :param allow_oversized: Whether you want to allow or not masks bigger than output image shape to exist
    :param verbose: 0 : No output, 1 : Errors, 2: Warnings, 3+: Info messages
    :param silent: Wheter to display progress or not
    :return: None
    """
    datasetPath = existing_path(datasetPath)
    outputDatasetPath = correct_path(outputDatasetPath)
    imageList = listdir2(datasetPath)
    if not silent:
        progressBar(0, len(imageList), prefix=f"Isolating {classToIsolate} class masks")
    for idx, (imageName, imageFolderPath) in enumerate(imageList):
        # Checking that the image has the class-to-isolate folder
        if not os.path.exists(os.path.join(imageFolderPath, classToIsolate)):
            continue
        # Listing the masks of the class to isolate, ignoring if none is present
        masksToIsolate = listdir2(os.path.join(imageFolderPath, classToIsolate))
        if len(masksToIsolate) > 0:
            inputClassFolderPath = os.path.join(datasetPath, imageName, '{className}')
            imagePath = os.path.join(inputClassFolderPath.format(className='images'), f'{imageName}.{imageFormat}')
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
            # For each class-to-isolate mask
            for maskToIsolateName, maskToIsolatePath in masksToIsolate:
                # Getting the mask id or giving it a unique one if failed
                try:
                    maskID = int(os.path.splitext(maskToIsolateName)[0].split('_')[1])
                except ValueError:
                    maskID = int(time())
                outputImageName = f'{imageName}_{maskID:03d}'
                outputFolderPath = os.path.join(outputDatasetPath, outputImageName, '{className}')

                # Loading mask to isolate and computing its bbox
                mask = loadSameResImage(maskToIsolatePath, image.shape)
                maskToIsolateBbox = extract_bboxes(mask)

                imageBbox = center_mask(maskToIsolateBbox, image.shape, image_size, verbose=verbose)
                maskToIsolate = mask[imageBbox[0]:imageBbox[2], imageBbox[1]:imageBbox[3]]
                if maskToIsolate.shape != (image_size, image_size):
                    if not allow_oversized:  # If mask does not fit into the wanted output shape and this is not allowed
                        if verbose > 0:
                            print(f"Skipping {outputImageName}: mask could not fit "
                                  "into {(image_size, image_size)} image")
                        continue  # Skipping the oversized mask
                    if verbose > 1:
                        print(f"{outputImageName} mask could not fit into {(image_size, image_size)} image")

                # Creating the output folder
                os.makedirs(outputFolderPath.format(className=classToIsolate), exist_ok=True)
                os.makedirs(outputFolderPath.format(className='images'), exist_ok=True)

                # Saving mask, cleaning image and saving it too
                cv2.imwrite(os.path.join(outputFolderPath.format(className=classToIsolate),
                                         f'{outputImageName}.{imageFormat}'), maskToIsolate, CV2_IMWRITE_PARAM)
                temp = image[imageBbox[0]:imageBbox[2], imageBbox[1]:imageBbox[3], :]
                temp = cv2.bitwise_and(temp, np.repeat(maskToIsolate[:, :, np.newaxis], 3, axis=2))
                cv2.imwrite(os.path.join(outputFolderPath.format(className='images'),
                                         f'{outputImageName}.{imageFormat}'), temp, CV2_IMWRITE_PARAM)

                # For each other folder (full_images and masks)
                for className, maskFolderPath in listdir2(imageFolderPath):
                    if os.path.isfile(maskFolderPath):
                        continue
                    elif className not in ['images', classToIsolate]:
                        if className == 'full_images':  # If folder is 'full_images' then only crop the image
                            temp = cv2.imread(os.path.join(maskFolderPath, f'{imageName}.{imageFormat}'),
                                              cv2.IMREAD_COLOR)
                            temp = temp[imageBbox[0]:imageBbox[2], imageBbox[1]:imageBbox[3], :]
                            os.makedirs(outputFolderPath.format(className='full_images'), exist_ok=True)
                            cv2.imwrite(os.path.join(outputFolderPath.format(className='full_images'),
                                                     f'{outputImageName}.{imageFormat}'),
                                        temp, CV2_IMWRITE_PARAM)
                        else:  # If folder is a mask folder, we have to test for each mask if it is inside the main mask
                            first = True
                            for currentMaskName, currentMaskPath in listdir2(maskFolderPath):
                                try:
                                    currentMaskID = int(os.path.splitext(currentMaskName)[0].split('_')[1])
                                except ValueError:
                                    currentMaskID = int(time())
                                temp = cv2.imread(currentMaskPath, cv2.IMREAD_GRAYSCALE)

                                # Extracting the bbox of the current mask to test if is inside the main mask
                                currentMaskBbox = extract_bboxes(temp)
                                # TODO Suggestion : check on a per-pixel approche as mask could be outside of the mask
                                #                   even if inside the RoI
                                if in_roi(currentMaskBbox, maskToIsolateBbox):
                                    if first:
                                        os.makedirs(outputFolderPath.format(className=className), exist_ok=True)
                                        first = False
                                    temp = temp[imageBbox[0]:imageBbox[2], imageBbox[1]:imageBbox[3]]
                                    cv2.imwrite(os.path.join(outputFolderPath.format(className=className),
                                                             f'{outputImageName}_{currentMaskID:03d}.{imageFormat}'),
                                                temp, CV2_IMWRITE_PARAM)
        if not silent:
            progressBar(idx + 1, len(imageList), prefix=f"Isolating {classToIsolate} class masks")
