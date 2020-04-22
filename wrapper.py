import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def createMask(imgName, imgShape, noMask, ptsMask, datasetName='dataset_train', maskClass='masks'):
    # Formatting the suffix for the image representing the mask
    maskName = str('0' if noMask < 100 else '') + str('0' if noMask < 10 else '') + str(noMask)
    # Defining path where the result image will be stored and creating dir if not exists
    output_directory = 'dataset_train/' + imgName + '/' + maskClass + '/'
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
    img = cv2.imread(directoryPath + '/' + imgName)
    if img is None:
        return
    shape = img.shape

    # Copying the original image in the dataset
    fileName = imgName.split('.')[0]
    targetDirectoryPath = datasetName + '/' + fileName + '/images/'
    if not os.path.exists(targetDirectoryPath):
        os.makedirs(targetDirectoryPath)
        cv2.imwrite(targetDirectoryPath + imgName, img)

    # https://www.datacamp.com/community/tutorials/python-xml-elementtree
    tree = ET.parse(directoryPath + '/' + fileName + '.xml')
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
        createMask(fileName, shape, int(noMask), ptsMask, datasetName, maskClass)


def startWrapper(rawDatasetPath):
	# Getting all images, assuming that there are only images and xml annotation in directory
	# mkyong.com/python/python-how-to-list-all-files-in-a-directory/
	images = []
	annotations = []
	for _, _, files in os.walk(rawDatasetPath):
		for file in files:
			if '.png' in file:
				images.append(file)
			elif '.xml' in file:
				annotations.append(file)

	numberOfImages = len(images)
	numberOfFiles = len(annotations)
	if numberOfFiles - numberOfImages != 0:
		print('It seems there are not as many images ({}) as annotations files ({})'.format(numberOfImages, numberOfFiles))
	for index in range(numberOfImages):
		file = images[index]
		print('Creating masks for {} image ({}/{})'.format(file, index + 1, numberOfImages))
		createMasksOfImage(rawDatasetPath, file)
