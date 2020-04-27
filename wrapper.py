import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def createMask(imgName, imgShape, noMask, ptsMask, datasetName='dataset_train', maskClass='masks'):
    # Formatting the suffix for the image representing the mask
    maskName = str('0' if noMask < 100 else '') + str('0' if noMask < 10 else '') + str(noMask)
    # Defining path where the result image will be stored and creating dir if not exists
    output_directory = datasetName + '/' + imgName + '/' + maskClass + '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # https://www.programcreek.com/python/example/89415/cv2.fillPoly
    # Formatting coordinates matrix to get int
    ptsMask = np.double(ptsMask)
    ptsMask = np.matrix.round(ptsMask)
    ptsMask = np.int32(ptsMask)

    # Creating black matrix with same size than original image and then drawing the mask
    mask = np.uint8(np.zeros((imgShape[0], imgShape[1])))
    cv2.fillPoly(mask, [ptsMask], 255)

    # Saving result image
    output_name = imgName + maskName + '.png'
    cv2.imwrite(output_directory + output_name, mask)


def createMasksOfImage(directoryPath, imgName, datasetName='dataset_train'):
    # Getting shape of original image (same for all this masks)
    img = None
    img = cv2.imread(directoryPath + '/' + imgName + '.png')
    if img is None:
        print('Problem with {} image'.format(imgName))
        return
    shape = img.shape

    # Copying the original image in the dataset
    targetDirectoryPath = datasetName + '/' + imgName + '/images/'
    if not os.path.exists(targetDirectoryPath):
        os.makedirs(targetDirectoryPath)
        cv2.imwrite(targetDirectoryPath + imgName + '.png', img)

    # https://www.datacamp.com/community/tutorials/python-xml-elementtree
    tree = ET.parse(directoryPath + '/' + imgName + '.xml')
    root = tree.getroot()
    # Going through the XML tree and getting all Annotation nodes
    for annotation in root.findall('./Annotations/Annotation'):
        maskClass = annotation.attrib.get('PartOfGroup')
        noMask = annotation.attrib.get('Name').replace('Annotation ', '')
        ptsMask = []
        # Going through the Annotation node and getting all Coordinate nodes
        for points in annotation.find('Coordinates'):
            xCoordinate = points.attrib.get('X')
            yCoordinate = points.attrib.get('Y')
            ptsMask.append([xCoordinate, yCoordinate])
        # print('Mask ' + noMask + ': NbPts = ' + str(len(ptsMask)) + '\tclass = ' + maskClass)
        createMask(imgName, shape, int(noMask), ptsMask, datasetName, maskClass)


def startWrapper(rawDatasetPath, datasetName='dataset_train'):
    # https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    names = []
    images = []  # list of image that can be used to compute masks
    missingImages = []  # list of missing images
    missingAnnotations = []  # list of missing annotations
    for _, _, files in os.walk(rawDatasetPath):
        for file in files:
            name = file.split('.')[0]
            if name not in names:  # We want to do this only once per unique file name (without extension)
                names.append(name)

                # Testing if there is an png image with that name
                pngPath = os.path.join(rawDatasetPath, name + '.png')
                pngExists = os.path.exists(pngPath)

                # Same thing with jp2 format
                jp2Path = os.path.join(rawDatasetPath, name + '.jp2')
                jp2Exists = os.path.exists(jp2Path)

                # Testing if annotation file exists for that name
                annotationsExist = os.path.exists(os.path.join(rawDatasetPath, name + '.xml'))
                if pngExists or jp2Exists:  # At least one image exists
                    if not annotationsExist:  # Annotations are missing
                        missingAnnotations.append(name)
                    else:
                        if not pngExists:  # Only jp2 exists
                            image = cv2.imread(jp2Path)  # Opening the jp2 image
                            cv2.imwrite(pngPath, image)  # Saving the jp2 image to png format
                        images.append(name)  # Adding this image to the list
                elif annotationsExist:  # There is no image file but xml found
                    missingImages.append(name)

    print()
    # Displaying missing image files
    nbMissingImg = len(missingImages)
    if nbMissingImg > 0:
        print('Missing {} image{} : {}'.format(nbMissingImg, 's' if nbMissingImg > 1 else '', missingImages))

    # Displaying missing annotations files
    nbMissingAnnotations = len(missingAnnotations)
    if nbMissingAnnotations > 0:
        print('Missing {} annotation{} : {}'.format(nbMissingAnnotations, 's' if nbMissingAnnotations > 1 else '',
                                                    missingAnnotations))

    # Checking if there is file that is not image nor annotation
    nbImages = len(images)
    if len(names) - nbMissingImg - nbMissingAnnotations - nbImages != 0:
        print('Be careful, there are not only required dataset files in this folder')

    # Creating masks for any image which has all required files and displaying progress
    for index in range(nbImages):
        file = images[index]
        print('Creating masks for {} image {}/{} ({:.2f}%)'.format(file, index + 1, nbImages, (index + 1) / nbImages * 100))
        createMasksOfImage(rawDatasetPath, file, datasetName)
