import os
import wrapper as wr
import datasetDivider as dd


def infoNephrologyDataset(datasetPath):
    print()
    histo = {}
    cortexMissing = []
    nbImg = 0
    for imageDir in os.listdir(datasetPath):
        nbImg += 1
        imagePath = os.path.join(datasetPath, imageDir)
        cortex = False
        for maskDir in os.listdir(imagePath):
            if maskDir == 'cortex':
                cortex = True
            if maskDir != "images":
                if maskDir not in histo:
                    histo[maskDir] = 1
                else:
                    histo[maskDir] += 1
        if not cortex:
            cortexMissing.append(imageDir)

    print("Nb Images : {}".format(nbImg))
    print("Histogram : {}".format(histo))
    print("Missing cortices : {}".format(cortexMissing))


wr.startWrapper('raw_dataset', 'temp_nephrology_dataset')
infoNephrologyDataset('temp_nephrology_dataset')
dd.divideDataset('temp_nephrology_dataset', 'nephrology_dataset', squareSideLength=1024)
# wr.startWrapper('raw_dataset_test', 'temp_test_dataset')
# infoNephrologyDataset('temp_test_dataset')
# dd.divideDataset('temp_test_dataset', 'test_dataset', squareSideLength=1024)
