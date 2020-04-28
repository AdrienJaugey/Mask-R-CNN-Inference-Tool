import os
import wrapper as wr
import datasetDivider as dd


def infoNephrologyDataset(datasetPath: str):
    """
    Print information about a dataset
    :param datasetPath: path to the dataset
    :return: None
    """
    print()
    histogram = {'tubule_atrophique': 0, 'vaisseau': 0, 'pac': 0, 'nsg_complet': 0,
                 'nsg_partiel': 0, 'tubule_sain': 0, 'cortex': 0}
    maxNbClasses = 0
    maxClasses = []
    maxNbClassesNoCortex = 0
    maxClassesNoCortex = []
    cortexMissing = []
    nbImg = 0
    for imageDir in os.listdir(datasetPath):
        nbImg += 1
        imagePath = os.path.join(datasetPath, imageDir)
        cortex = False
        localHisto = {'tubule_atrophique': 0, 'vaisseau': 0, 'pac': 0, 'nsg_complet': 0,
                      'nsg_partiel': 0, 'tubule_sain': 0, 'cortex': 0}
        for maskDir in os.listdir(imagePath):
            if maskDir == 'cortex':
                cortex = True
            if maskDir != "images":
                histogram[maskDir] += 1
                localHisto[maskDir] += 1
        if not cortex:
            cortexMissing.append(imageDir)

        nbClasses = 0
        nbClassesNoCortex = 0
        for objectClass in localHisto:
            if localHisto[objectClass] > 0:
                if objectClass != 'cortex':
                    nbClassesNoCortex += 1
                nbClasses += 1
        if nbClasses >= maxNbClasses:
            if nbClasses > maxNbClasses:
                maxNbClasses = nbClasses
                maxClasses = []
            maxClasses.append(imageDir)

        if nbClassesNoCortex >= maxNbClassesNoCortex:
            if nbClassesNoCortex > maxNbClassesNoCortex:
                maxNbClassesNoCortex = nbClassesNoCortex
                maxClassesNoCortex = []
            maxClassesNoCortex.append(imageDir)

    print("Nb Images : {}".format(nbImg))
    print("Histogram : {}".format(histogram))
    print("Missing cortices : {}".format(cortexMissing))
    print("Max Classes ({}) : {}".format(maxNbClasses, maxClasses))
    print("Max Classes w/o cortex (({}) : {}".format(maxNbClassesNoCortex, maxClassesNoCortex))


wr.startWrapper('raw_dataset', 'temp_nephrology_dataset')
infoNephrologyDataset('temp_nephrology_dataset')
dd.divideDataset('temp_nephrology_dataset', 'nephrology_dataset', squareSideLength=1024)
infoNephrologyDataset('nephrology_dataset')
# wr.startWrapper('raw_dataset_test', 'temp_test_dataset')
# infoNephrologyDataset('temp_test_dataset')
# dd.divideDataset('temp_test_dataset', 'test_dataset', squareSideLength=1024)
# infoNephrologyDataset('test_dataset')
