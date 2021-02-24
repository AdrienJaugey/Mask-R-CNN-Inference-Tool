import math
import os
from time import time

import cv2
import numpy as np

from common_utils import progressBar, formatTime

VERBOSE = False
CV2_IMWRITE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 100, cv2.IMWRITE_PNG_COMPRESSION, 9]

def computeStartsOfInterval(maxVal: int, intervalLength=1024, min_overlap_part=0.33):
    """
    Divide the [0; maxVal] interval into a uniform distribution with at least min_overlap_part of overlapping
    :param maxVal: end of the base interval
    :param intervalLength: length of the new intervals
    :param min_overlap_part: min overlapping part of intervals, if less, adds intervals with length / 2 offset
    :return: list of starting coordinates for the new intervals
    """
    nbDiv = math.ceil(maxVal / intervalLength)
    # Computing gap to get something that tends to a uniform distribution
    gap = (nbDiv * intervalLength - maxVal) / (nbDiv - 1)
    coordinates = []
    for i in range(nbDiv):
        coordinate = round(i * (intervalLength - gap))
        if i == nbDiv - 1:
            # Should not be useful but acting as a security
            coordinates.append(maxVal - intervalLength)
        else:
            coordinates.append(coordinate)
            # If gap is not enough, we add division with a intervalLength / 2 offset
            if gap < intervalLength * min_overlap_part:
                coordinates.append(coordinate + intervalLength // 2)
    return coordinates


def getMaxSizeForDivAmount(divAmount: int, intervalLength=1024, min_overlap_part=0.33):
    res = intervalLength * int(2. - min_overlap_part)
    while len(computeStartsOfInterval(res + 1, intervalLength, min_overlap_part)) <= divAmount:
        res += 1
    return res


def getDivisionsCount(xStarts: [int], yStarts: [int]):
    """
    Return the number of division for given starting x and y coordinates
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :return: number of divisions
    """
    return len(xStarts) * len(yStarts)


def getDivisionByID(xStarts: [int], yStarts: [int], idDivision: int, divisionSize=1024):
    """
    Return x and y starting and ending coordinates for a specific division
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want the coordinates. 0 <= ID < number of divisions
    :param divisionSize: length of the new intervals
    :return: x, xEnd, y, yEnd coordinates
    """
    # assert divisionSize is int or divisionSize is tuple
    if not 0 <= idDivision < len(xStarts) * len(yStarts):
        return -1, -1, -1, -1
    yIndex = idDivision // len(xStarts)
    xIndex = idDivision - yIndex * len(xStarts)

    x = xStarts[xIndex]
    xEnd = x + (divisionSize if type(divisionSize) is int else divisionSize[0])

    y = yStarts[yIndex]
    yEnd = y + (divisionSize if type(divisionSize) is int else divisionSize[1])
    return x, xEnd, y, yEnd


def getDivisionID(xStarts: [int], yStarts: [int], xStart: int, yStart: int):
    """
    Return the ID of the dimension from its starting x and y coordinates
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param xStart: the x starting coordinate of the division
    :param yStart: the y starting coordinate of the division
    :return: the ID of the division. 0 <= ID < number of divisions
    """
    xIndex = xStarts.index(xStart)
    yIndex = yStarts.index(yStart)
    return yIndex * len(xStarts) + xIndex


def getImageDivision(img, xStarts: [int], yStarts: [int], idDivision: int, divisionSize=1024):
    """
    Return the wanted division of an Image
    :param img: the base image
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want to get. 0 <= ID < number of divisions
    :param divisionSize: length of division side
    :return: the image division
    """
    # assert divisionSize is int or divisionSize is tuple
    x, xEnd, y, yEnd = getDivisionByID(xStarts, yStarts, idDivision, divisionSize)
    if len(img.shape) == 2:
        return img[y:yEnd, x:xEnd]
    else:
        return img[y:yEnd, x:xEnd, :]


def getBWCount(mask):
    """
    Return number of black (0) and white (>0) pixels in a mask image
    :param mask: the mask image
    :return: number of black pixels, number of white pixels
    """
    mask = mask.astype(np.bool).flatten()
    totalPx = int(mask.shape[0])
    whitePx = int(np.sum(mask))
    return totalPx - whitePx, whitePx


def getRepresentativePercentage(blackMask: int, whiteMask: int, divisionImage):
    """
    Return the part of area represented by the white pixels in division, in mask and in image
    :param blackMask: number of black pixels in the base mask image
    :param whiteMask: number of white pixels in the base mask image
    :param divisionImage: the mask division image
    :return: partOfDiv, partOfMask, partOfImage
    """
    blackDiv, whiteDiv = getBWCount(divisionImage)
    partOfDiv = whiteDiv / (whiteDiv + blackDiv) * 100
    partOfMask = whiteDiv / whiteMask * 100
    partOfImage = whiteDiv / (blackMask + whiteMask) * 100
    return partOfDiv, partOfMask, partOfImage


def divideDataset(inputDatasetPath: str, outputDatasetPath: str = None, squareSideLength=1024, min_overlap_part=0.33,
                  min_part_of_div=10.0, min_part_of_cortex=10.0, min_part_of_mask=10.0, mode: str = "main", verbose=0):
    """
    Divide a dataset using images bigger than a wanted size into equivalent dataset with square-images divisions of
    the wanted size
    :param inputDatasetPath: path to the base dataset to divide
    :param outputDatasetPath: path to the output divided dataset
    :param squareSideLength: length of a division side
    :param min_overlap_part: min overlapping part of intervals, if less, adds intervals with length / 2 offset
    :param min_part_of_div: min part of div, used to decide whether the div will be used or not
    :param min_part_of_cortex: min part of cortex, used to decide whether the div will be used or not
    :param min_part_of_mask: min part of mask, used to decide whether the div will be used or not
    :param mode: Whether it is main or cortex dataset
    :param verbose:
    :return: None
    """
    if outputDatasetPath is None:
        outputDatasetPath = inputDatasetPath + '_divided'
    print()

    imageList = os.listdir(inputDatasetPath)
    start_time = time()
    for idx, imageDir in enumerate(imageList):
        if verbose > 0:
            progressBar(idx, len(imageList), prefix="Dividing dataset ",
                        suffix=f" {formatTime(time() - start_time)} Current : {imageDir}")
        imageDirPath = os.path.join(inputDatasetPath, imageDir)

        imagePath = os.path.join(imageDirPath, 'images')
        imagePath = os.path.join(imagePath, os.listdir(imagePath)[0])
        imageFormat = os.path.basename(imagePath).split('.')[-1]

        image = cv2.imread(imagePath)
        height, width, _ = image.shape

        xStarts = computeStartsOfInterval(width, squareSideLength, min_overlap_part)
        yStarts = computeStartsOfInterval(height, squareSideLength, min_overlap_part)

        '''###################################
        ### Exclusion of Useless Divisions ###
        ###################################'''
        cortexDirPath = os.path.join(imageDirPath, 'cortex')
        isThereCortex = os.path.exists(cortexDirPath)
        excludedDivisions = []
        tryCleaning = False
        if not isThereCortex:
            tryCleaning = True
        else:
            cortexImgPath = os.path.join(cortexDirPath, os.listdir(cortexDirPath)[0])
            usefulPart = cv2.imread(cortexImgPath, cv2.IMREAD_UNCHANGED)
            if mode == "cortex":
                for folder in os.listdir(imageDirPath):
                    if folder not in ['images', 'fullimages']:
                        maskDir = os.path.join(imageDirPath, folder)
                        for mask in os.listdir(maskDir):
                            mask = cv2.imread(os.path.join(maskDir, mask), cv2.IMREAD_UNCHANGED)
                            usefulPart = cv2.bitwise_or(usefulPart, mask)
            black, white = getBWCount(usefulPart)
            for divId in range(getDivisionsCount(xStarts, yStarts)):
                div = getImageDivision(usefulPart, xStarts, yStarts, divId, squareSideLength)
                partOfDiv, partOfCortex, partOfImage = getRepresentativePercentage(black, white, div)
                excluded = partOfDiv < min_part_of_div or partOfCortex < min_part_of_cortex
                if excluded:
                    excludedDivisions.append(divId)

        '''####################################################
        ### Creating output dataset files for current image ###
        ####################################################'''
        imageOutputDirPath = os.path.join(outputDatasetPath, imageDir)
        for masksDir in os.listdir(imageDirPath):
            if masksDir in ['images', 'full_images']:
                for divId in range(getDivisionsCount(xStarts, yStarts)):
                    if divId in excludedDivisions:
                        continue
                    divisionOutputDirPath = f"{imageOutputDirPath}_{divId:02d}"
                    outputImagePath = os.path.join(divisionOutputDirPath, masksDir)
                    os.makedirs(outputImagePath, exist_ok=True)
                    outputImagePath = os.path.join(outputImagePath, f"{imageDir}_{divId:02d}.{imageFormat}")
                    tempPath = os.path.join(imageDirPath, masksDir, f'{imageDir}.{imageFormat}')
                    tempImage = cv2.imread(tempPath)
                    cv2.imwrite(outputImagePath, getImageDivision(tempImage, xStarts, yStarts, divId, squareSideLength),
                                CV2_IMWRITE_PARAM)
            else:
                maskDirPath = os.path.join(imageDirPath, masksDir)
                # Going through every mask file in current directory
                for mask in os.listdir(maskDirPath):
                    maskPath = os.path.join(maskDirPath, mask)
                    maskImage = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
                    blackMask, whiteMask = getBWCount(maskImage)

                    # Checking for each division if mask is useful or not
                    for divId in range(getDivisionsCount(xStarts, yStarts)):
                        if divId in excludedDivisions:
                            continue
                        divMaskImage = getImageDivision(maskImage, xStarts, yStarts, divId, squareSideLength)

                        _, partOfMask, _ = getRepresentativePercentage(blackMask, whiteMask, divMaskImage)
                        if partOfMask >= min_part_of_mask:
                            divisionOutputDirPath = f"{imageOutputDirPath}_{divId:02d}"
                            maskOutputDirPath = os.path.join(divisionOutputDirPath, masksDir)
                            os.makedirs(maskOutputDirPath, exist_ok=True)
                            outputMaskPath = os.path.join(maskOutputDirPath, f"{mask.split('.')[0]}_{divId:02d}.{imageFormat}")
                            cv2.imwrite(outputMaskPath, divMaskImage, CV2_IMWRITE_PARAM)

        if tryCleaning:
            for divId in range(getDivisionsCount(xStarts, yStarts)):
                divisionOutputDirPath = f"{imageOutputDirPath}_{divId:02d}"
                if len(os.listdir(divisionOutputDirPath)) == 1:
                    imagesDirPath = os.path.join(divisionOutputDirPath, 'images')
                    imageDivPath = os.path.join(imagesDirPath, f"{imageDir}_{divId:02d}.{imageFormat}")
                    os.remove(imageDivPath)  # Removing the .png image in /images
                    os.removedirs(imagesDirPath)  # Removing the /images folder
    if verbose > 0:
        progressBar(1, 1, prefix="Dividing dataset ", suffix=f" {formatTime(time() - start_time)}" + " " * 20)
