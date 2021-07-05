# Index
1. [Introduction](#1-introduction)
2. [Supported formats](#2-supported-formats)
3. [Adding another annotations format](#3-adding-another-annotations-format)
   * [Custom Adapter Class Template](#custom-adapter-class-template)
   * [Making your custom Adapter Class available automatically](#making-your-custom-adapter-class-available-automatically)

# 1. Introduction
To train Mask R-CNN, you need to get annotations for your dataset images. As we based our model on [navidyou's Mask R-CNN implementation for cell nucleus detection](https://github.com/navidyou/Mask-RCNN-implementation-for-cell-nucleus-detection-executable-on-google-colab-), we used the same mask loading system. Instead of a masks folder where to get every mask as an image, we are using a folder per class containing all its corresponding masks. You can generate your dataset using the [datasetFormator](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/datasetFormator.py) script which requires all the [datasetTools](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/tree/master/datasetTools) folder's files. 

If you want to be able to read directly the annotations file from Mask R-CNN, without generating an image per mask, you will need to rewrite a custom dataset Class. You should read [Waleed Abdulla's article](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) on Mask R-CNN where he explains how to do this.

Annotations files are also generated from predictions, so you can generate new data for future training. However, you may have to fix some of the predicted annotations.

To sum up, to be able to train using a specific annotation format, the model needs to read and write this format. This work is done by adapter classes.

# 2. Supported formats
Current supported annotations format are the following :
* ASAP format ([ASAPAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/ASAPAdapter.py))
* Label Me format ([LabelMeAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/LabelMeAdapter.py))

# 3. Adding another annotations format
You can add support to a currently unsupported annotations format by writing an adapter class inheriting from abstract classes [AnnotationAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/AnnotationAdapter.py#L7), [XMLAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/AnnotationAdapter.py#L87), or [JSONAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/AnnotationAdapter.py#L118) and implementing all abstract methods.  
_You can use [ASAPAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/ASAPAdapter.py) or [LabelMeAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/LabelMeAdapter.py) as examples to help you._

## Custom Adapter Class Template
Here is the custom adapter template :
```python
class CustomAdapter(JSONAdapter):
    '''
    In this example, this adapter would create and read JSON files. You can also 
    inherits from XMLAdapter if using XML files. If you are using an unique 
    annotations format, that is not used for anything else (not a basic extension
    for example), you should inherits directly from AnnotationAdapter and also 
    overwrite __str__(self), getSaveFileName(self, fileName) and 
    getAnnotationFormat().
    '''

    def __init__(self, imageInfo: dict, verbose=0):
        super().__init__(imageInfo, verbose=verbose)
        '''
        Annotations format can have information about the version of the format,
        about the image itself (height, width, name...) that are present only 
        once in the file. This information should be formatted and added here.

        imageInfo parameter contains at least "name", "height" and "width" values
        for the current image. To access to information, use imageInfo['name'] 
        for example
        '''

    def addAnnotation(self, classInfo: {}, points):
        '''
        Here is the method adding an annotation, this is where you have to
        format the data to get the correct annotations format. Using the
        same keywords and structures as you custom format.

        classInfo parameter contains the name, id and other information you
        may have added for the given annotation/mask. For example, use
        classInfo['name'] to get access to the class name of the current mask
        '''
        pass

    def addAnnotationClass(self, classInfo: {}):
        '''
        Some annotations formats are saving informations about each detection
        class. You should implement that part here. If nothing has to be done
        keep the method and use the pass keyword as done here.

        You can access to class information with classInfo parameter. For 
        example, to get access to the class ID, use classInfo['id']
        '''
        pass

    def saveToFile(self, savePath, fileName):
        '''
        If you have specific tasks to realize before saving, you should do them 
        here.
        '''
        super().saveToFile(savePath, fileName)
        '''
        Except the saving is different than just writing the data 
        (object.__str__()) into a file, you may just call 
        super().saveToFile() has written before.
        '''

    @staticmethod
    def getPriorityLevel():
        '''
        Return an integer here representing the priority level of the adapter.
        When datasetWrapper has to create the masks of an image and it has not
        received any Adapter class in parameters, it will automatically choose
        the Adapter class that can read one of all annotations files available
        with the highest priority level.

        Example :
        There are a Label Me Annotations file and an ASAP Annotations file for
        the same image when you started the inference. There is no choice for a
        specific adapter. The Wrapper will check for compatible annotations files
        extensions (so .xml and .json). For each file, it will test each Adapter
        and it will keep one file with an Adapter that can read it. This choice
        is made using the priority level. If two adapters are available, the one
        with the highest priority level.
        '''

    @staticmethod
    def canRead(filePath):
        canRead = JSONAdapter.canRead(filePath)
        if canRead:
            '''
            You can validate the json file to check it has the schema/structure 
            you want
            '''
            with open(filePath, 'r') as file:
                data = json.load(file)
                try:
                    sch.validate(instance=data, schema=CUSTOM_FORMAT_SCHEMA)
                    canRead = True
                except sch.exceptions.ValidationError as err:
                    canRead = False
            '''
            You can also check by other methods if it is a file is readable
            '''
        return canRead

    @staticmethod
    def readFile(filePath):
        canRead = CustomAdapter.canRead(filePath)
        assert canRead
        masks = []
        with open(filePath, 'r') as file:
            '''
            Here is the part where you get all annotations information saved in the 
            file.
            For each annotation, append a tuple (classIdorName, [[x, y]]) to masks.
            '''
        return masks
```

## Making your custom Adapter Class available automatically
Finally, you should add your custom Adapter class at the end of the [AnnotationAdapter](https://github.com/AdrienJaugey/Custom-Mask-R-CNN-for-kidney-s-cell-recognition/blob/master/datasetTools/AnnotationAdapter.py#L142) script. You just have to import the class (do it at the end like for `ASAPAdapter` and `LabelMeAdapter`) and adding it to `ANNOTATION_ADAPTERS` list.

Here is an example of what it should be with the `CustomAdapter` class :
```python
from datasetTools.ASAPAdapter import ASAPAdapter
from datasetTools.LabelMeAdapter import LabelMeAdapter
from datasetTools.CustomAdapter import CustomAdapter

ANNOTATION_ADAPTERS = [ASAPAdapter, LabelMeAdapter, CustomAdapter]
```