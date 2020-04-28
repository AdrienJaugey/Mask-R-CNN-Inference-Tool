import os
import cv2

VERBOSE = False
MIN_PART_OF_DIV = 10.0
MIN_PART_OF_CORTEX = 10.0
MIN_PART_OF_MASK = 10.0


def computeStartsOfInterval(maxVal: int, intervalLength=1024):
    """
    Divide the [0; maxVal] interval into the least overlapping (or not) intervals of given length
    :param maxVal: end of the base interval
    :param intervalLength: length of the new intervals
    :return: list of starting coordinates for the new intervals
    """
    nbIteration = maxVal // intervalLength
    assert nbIteration > 1
    # First interval always starts on coordinate 0
    startCoordinates = [0]
    if maxVal / intervalLength != nbIteration:
        if VERBOSE:
            print('{} is not a multiplier of {}'.format(intervalLength, maxVal))

        # offset = ((nbIteration - 1) * intervalLength - (num - 2 * intervalLength)) / 2  # Ok but no middle overlaps
        offset = ((nbIteration - 1) * intervalLength - (maxVal - 2 * intervalLength)) / (nbIteration + 1)
        nbIteration -= 1  # There are (num // intervalLength) + 1 intervals but the first and last are already counted
        for i in range(nbIteration):
            startCoordinates.append(round((i + 1) * (intervalLength - offset)))

        # Last interval
        startCoordinates.append(maxVal - intervalLength)
    else:
        if VERBOSE:
            print('{} is a multiplier of {}'.format(intervalLength, maxVal))
        for i in range(1, nbIteration):
            startCoordinates.append(i * intervalLength)
    return startCoordinates


def getDivisionsCount(xStarts: [int], yStarts: [int]):
    """
    Return the number of division for given starting x and y coordinates
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :return: number of divisions
    """
    return len(xStarts) * len(yStarts)


def getDivisionByID(xStarts: [int], yStarts: [int], idDivision: int, squareSideLength=1024):
    """
    Return x and y starting and ending coordinates for a specific division
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want the coordinates. 0 <= ID < number of divisions
    :param squareSideLength: length of the new intervals
    :return: x, xEnd, y, yEnd coordinates
    """
    if not 0 <= idDivision < len(xStarts) * len(yStarts):
        return -1, -1, -1, -1
    yIndex = idDivision // len(xStarts)
    xIndex = idDivision - yIndex * len(xStarts)

    x = xStarts[xIndex]
    xEnd = x + squareSideLength

    y = yStarts[yIndex]
    yEnd = y + squareSideLength
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


def getImageDivision(img, xStarts: [int], yStarts: [int], idDivision: int, squareSideLength=1024):
    """
    Return the wanted division of an Image
    :param img: the base image
    :param xStarts: the x-axis starting coordinates
    :param yStarts: the y-axis starting coordinates
    :param idDivision: the ID of the division you want to get. 0 <= ID < number of divisions
    :param squareSideLength: length of division side
    :return: the image division
    """
    x, xEnd, y, yEnd = getDivisionByID(xStarts, yStarts, idDivision, squareSideLength)
    if len(img.shape) == 2:
        return img[y:yEnd, x:xEnd]
    else:
        return img[y:yEnd, x:xEnd, :]


def getBWCount(mask):
    """
    Return number of black (0) and white (255) pixels in a mask image
    :param mask: the mask image
    :return: number of black pixels, number of white pixels
    """
    histogram = cv2.calcHist([mask], [0], None, [256], [0, 256])
    return int(histogram[0]), int(histogram[255])


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


def divideDataset(inputDatasetPath: str, outputDatasetPath: str = None, squareSideLength=1024):
    """
    Divide a dataset using images bigger than a wanted size into equivalent dataset with square-images divisions of
    the wanted size
    :param inputDatasetPath: path to the base dataset to divide
    :param outputDatasetPath: path to the output divided dataset
    :param squareSideLength: length of a division side
    :return: None
    """
    if outputDatasetPath is None:
        outputDatasetPath = inputDatasetPath + '_divided'
    print()

    nbImages = len(os.listdir(inputDatasetPath))
    iterator = 1
    for imageDir in os.listdir(inputDatasetPath):
        print("Dividing {} image {}/{} ({:.2f}%)".format(imageDir, iterator, nbImages, iterator / nbImages * 100))
        imageDirPath = os.path.join(inputDatasetPath, imageDir)

        imagePath = os.path.join(imageDirPath, 'images/{}.png'.format(imageDir))
        image = cv2.imread(imagePath)
        height, width, _ = image.shape

        xStarts = computeStartsOfInterval(width, squareSideLength)
        yStarts = computeStartsOfInterval(height, squareSideLength)

        '''###################################
        ### Exclusion of Useless Divisions ###
        ###################################'''
        cortexDirPath = os.path.join(imageDirPath, 'cortex')
        isThereCortex = os.path.exists(cortexDirPath)
        excludedDivisions = []
        tryCleaning = False
        if not isThereCortex:
            tryCleaning = True
            if VERBOSE:
                print("{} : No cortex".format(imageDir))
        else:
            cortexImgPath = os.path.join(cortexDirPath, os.listdir(cortexDirPath)[0])
            cortexImg = cv2.imread(cortexImgPath, cv2.IMREAD_UNCHANGED)
            # 8 355 840 pixels
            black, white = getBWCount(cortexImg)
            total = white + black
            if VERBOSE:
                print("Cortex is {:.3f}% of base image".format(white / total * 100))
                print("\t[ID] : Part of Div | of Cortex | of Image")
            for divId in range(getDivisionsCount(xStarts, yStarts)):
                div = getImageDivision(cortexImg, xStarts, yStarts, divId, squareSideLength)
                partOfDiv, partOfCortex, partOfImage = getRepresentativePercentage(black, white, div)
                excluded = partOfDiv < MIN_PART_OF_DIV or partOfCortex < MIN_PART_OF_CORTEX
                if excluded:
                    excludedDivisions.append(divId)
                if VERBOSE and partOfDiv != 0:
                    print("\t[{}{}] : {:.3f}% | {:.3f}% | {:.3f}% {}".format('0' if divId < 10 else '', divId,
                                                                             partOfDiv, partOfCortex, partOfImage,
                                                                             'EXCLUDED' if excluded else ''))
            if VERBOSE:
                print("\tExcluded Divisions : {} \n".format(excludedDivisions))

        '''####################################################
        ### Creating output dataset files for current image ###
        ####################################################'''
        imageOutputDirPath = os.path.join(outputDatasetPath, imageDir)
        for masksDir in os.listdir(imageDirPath):
            if masksDir == 'images':
                for divId in range(getDivisionsCount(xStarts, yStarts)):
                    if divId in excludedDivisions:
                        continue
                    divSuffix = "_{}{}".format('0' if divId < 10 else '', divId)
                    divisionOutputDirPath = imageOutputDirPath + divSuffix
                    outputImagePath = os.path.join(divisionOutputDirPath, 'images')
                    os.makedirs(outputImagePath, exist_ok=True)
                    outputImagePath = os.path.join(outputImagePath, imageDir + divSuffix + ".png")
                    cv2.imwrite(outputImagePath, getImageDivision(image, xStarts, yStarts, divId, squareSideLength))
            else:
                maskDirPath = os.path.join(imageDirPath, masksDir)
                # Going through every mask file in current directory
                for mask in os.listdir(maskDirPath):
                    maskPath = os.path.join(maskDirPath, mask)
                    maskImage = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
                    blackMask, whiteMask = getBWCount(maskImage)

                    # Checking for each division if mask is useful or not
                    for divId in range(getDivisionsCount(xStarts, yStarts)):
                        if divId in excludedDivisions:
                            continue
                        divMaskImage = getImageDivision(maskImage, xStarts, yStarts, divId, squareSideLength)

                        _, partOfMask, _ = getRepresentativePercentage(blackMask, whiteMask, divMaskImage)
                        if partOfMask >= MIN_PART_OF_MASK:
                            divSuffix = "_{}{}".format('0' if divId < 10 else '', divId)
                            divisionOutputDirPath = imageOutputDirPath + divSuffix
                            maskOutputDirPath = os.path.join(divisionOutputDirPath, masksDir)
                            os.makedirs(maskOutputDirPath, exist_ok=True)
                            outputMaskPath = os.path.join(maskOutputDirPath, mask.split('.')[0] + divSuffix + ".png")
                            cv2.imwrite(outputMaskPath, divMaskImage)

        if tryCleaning:
            for divId in range(getDivisionsCount(xStarts, yStarts)):
                divSuffix = "_{}{}".format('0' if divId < 10 else '', divId)
                divisionOutputDirPath = imageOutputDirPath + divSuffix
                if len(os.listdir(divisionOutputDirPath)) == 1:
                    imagesDirPath = os.path.join(divisionOutputDirPath, 'images')
                    imageDivPath = os.path.join(imagesDirPath, imageDir + divSuffix + '.png')
                    os.remove(imageDivPath)  # Removing the .png image in /images
                    os.removedirs(imagesDirPath)  # Removing the /images folder
        iterator += 1
