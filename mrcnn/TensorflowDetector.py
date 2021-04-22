import os
import json
import warnings
import cv2
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def load_image_into_numpy_array(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def reframe_box_masks_to_image_masks(box_masks, boxes, image_height, image_width, resize_method='bilinear'):
    """Transforms the box masks back to full image masks.
        src : github.com/tensorflow/models/research/object_detection/utils/ops.py
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.

  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.

  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
    resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method

    # TODO(rathodv): Make this a public function.
    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            denom = max_corner - min_corner
            # Prevent a divide by zero.
            denom = tf.math.maximum(denom, 1e-4)
            transformed_boxes = (boxes - min_corner) / denom
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

        # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
        # is static. This will help us run and test on TPUs.
        resized_crops = tf.image.crop_and_resize(
            image=box_masks_expanded,
            boxes=reverse_boxes,
            box_indices=tf.range(num_boxes),
            crop_size=[image_height, image_width],
            method=resize_method,
            extrapolation_value=0)
        return tf.cast(resized_crops, box_masks.dtype)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
    return tf.squeeze(image_masks, axis=3)


class TensorflowDetector:
    """
    Encapsulation of a Tensorflow saved model to facilitate inferences and visualisation
    Based on : https://bit.ly/3p2iPDc
    """

    def __init__(self, savedModelPath: str = None, labelMapPath: str = None):
        self.__CATEGORY_INDEX__ = None
        self.__MODEL__ = None
        self.__MODEL_PATH__ = None
        if savedModelPath is not None and os.path.exists(savedModelPath):
            self.load(savedModelPath, labelMapPath)

    def isLoaded(self):
        return self.__MODEL__ is not None

    def getModelPath(self):
        return self.__MODEL_PATH__

    def getConfig(self):
        return self.__CATEGORY_INDEX__

    def load(self, modelPath: str, labelMapPath: str = None):
        isExportedModelDir = os.path.exists(os.path.join(modelPath, 'saved_model'))
        isSavedModelDir = (os.path.exists(os.path.join(modelPath, 'saved_model.pb'))
                           and os.path.exists(os.path.join(modelPath, 'variables')))
        if not (isExportedModelDir or isSavedModelDir):
            raise ValueError("Provided model path is not a TF exported model or TF SavedModel folder.")
        if labelMapPath is None:
            if isExportedModelDir:
                labelMapPath = os.path.join(modelPath, "label_map.json")
            else:
                labelMapPath = os.path.join(modelPath, "assets", "label_map.json")

        with open(labelMapPath, 'r') as file:
            self.__CATEGORY_INDEX__ = {int(key): value for key, value in json.load(file).items()}
        print("Loading... ", end="")
        self.__MODEL__ = tf.saved_model.load(modelPath if isSavedModelDir else os.path.join(modelPath, 'saved_model'))
        self.__MODEL_PATH__ = modelPath
        print("Done !")

    def process(self, image, computeMaskBbox=False, normalizedCoordinates=True, score_threshold=None):
        """
        Run inference on an image
        :param image: path to the image to use or the image itself
        :param computeMaskBbox: Whether to return predicted bboxes or, if masks are predicted, compute boxes from them
        :param normalizedCoordinates: Whether to use normalized coordinates or not
        :param score_threshold: minimal score of predictions
        :return: The image and refactored dict from the inference
        """
        if not self.isLoaded():
            raise RuntimeError(f"Detector is not loaded, please use {self.__class__.__qualname__}::load() with correct "
                               f"parameters to load a model before calling {self.__class__.__qualname__}::process().")
        # Load image into a tensor
        if image is None:
            raise ValueError("Image cannot be None value")
        if type(image) is str:
            image_np = load_image_into_numpy_array(image)
        elif type(image) is np.ndarray:
            image_np = image
        else:
            raise ValueError("Image parameter must be either path as str or image as ndarray")
        height, width = image_np.shape[:2]
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Running the inference
        detections = self.__MODEL__(input_tensor)

        # Getting the number of detection (remove batch size from the original result)
        num_detections = int(detections.pop('num_detections'))

        # Removing low scores predictions if threshold given
        scores = detections['detection_scores'][0, :num_detections].numpy()
        if score_threshold is not None:
            low_score_idx = np.where(scores < score_threshold)[0]
            if low_score_idx.size > 0:
                num_detections = low_score_idx[0]
                scores = scores[:num_detections]
        results = {'scores': scores}

        # Converting output format of the TF OD model to the format used in our code
        keysEquivalent = {'detection_boxes': 'rois', 'detection_classes': 'class_ids'}
        for key, outputkey in keysEquivalent.items():
            if key in detections:
                results[outputkey] = detections[key][0, :num_detections].numpy()
        results['class_ids'].astype(int)
        results['rois'].astype(float)

        # Converting coordinates if needed
        if not normalizedCoordinates:
            results['rois'] = np.around(results['rois'] * np.array([height, width, height, width])).astype(int)

        # Converting masks to binary (uint8 0 or 255 actually) masks
        if 'detection_masks' in detections:
            results['masks'] = reframe_box_masks_to_image_masks(
                detections['detection_masks'][0, :num_detections],
                detections['detection_boxes'][0, :num_detections],
                image_height=height, image_width=width
            )
            results['masks'] = tf.cast(tf.greater(results['masks'], 0.5), tf.uint8).numpy()
            # Convert (N, H, W) to (H, W, N)
            results['masks'] = np.moveaxis(results['masks'], 0, -1) * 255

            # Computing bounding boxes if needed
            if computeMaskBbox:
                from mrcnn.utils import extract_bboxes
                results['rois'] = extract_bboxes(results['masks'])

        return results

    '''def applyResults(self, image, results: dict, maxObjectToDraw=200, minScoreThreshold=0.3, drawImage=True,
                     mapText=False, low_memory=False):
        """
        Draw results on the image
        :param image: the image to draw the results on
        :param results: the detection results of the image
        :param maxObjectToDraw: maximum number of object to draw on the image
        :param minScoreThreshold: minimum score of the boxes to draw
        :param drawImage: if True, will draw predictions on the image
        :param mapText: if True, will return the content of the map txt file as a string
        :param low_memory: if True, will replace the input image
        :return: The image with results if enabled else None, the map file as string if enabled
        """
        image_with_detections = None
        if drawImage:
            image_with_detections = image if low_memory else image.copy()
        resMapText = "" if mapText else None

        if drawImage and 'detection_masks' in results:
            for idx in range(min(results['num_detections'], maxObjectToDraw)):
                if results['detection_scores'][idx] < minScoreThreshold:
                    break
                image_with_detections = apply_mask(
                    image=image_with_detections,
                    mask=results['detection_masks'][:, :, idx],
                    color=self.__CATEGORY_INDEX__[results['detection_classes'][idx]]["color"],
                    bbox=results['detection_boxes'][idx, :]
                )
            image_with_detections = cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR)
        for idx in range(min(results['num_detections'], maxObjectToDraw)):
            score = results['detection_scores'][idx]
            if results['detection_scores'][idx] < minScoreThreshold:
                break
            height, width, _ = image.shape
            yMin, xMin, yMax, xMax = tuple(results['detection_boxes'][idx, :])
            if type(yMin) is float:
                yMin = int(yMin * height)
                xMin = int(xMin * width)
                yMax = int(yMax * height)
                xMax = int(xMax * width)
            classId = results['detection_classes'][idx]
            className = self.__CATEGORY_INDEX__[classId]["name"]
            if mapText:
                # <class_name> <confidence> <left> <top> <right> <bottom>
                resMapText += "{}{} {:.6f} {} {} {} {}".format("" if resMapText == "" else "\n", className, score,
                                                               xMin, yMin, xMax, yMax)
            if drawImage:
                # Convert color from RGB to BGR
                color = tuple(self.__CATEGORY_INDEX__[classId]["color"][::-1])
                image_with_detections = cv2.rectangle(image_with_detections, (xMin, yMin), (xMax, yMax), color, 3)
                scoreText = '{}: {:.0%}'.format(className, score)
                image_with_detections[yMin - 12:yMin + 2, xMin:xMin + 9 * len(scoreText), :] = color
                image_with_detections = cv2.putText(image_with_detections, scoreText, (xMin, yMin),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        image_with_detections = cv2.cvtColor(image_with_detections, cv2.COLOR_BGR2RGB)
        return image_with_detections, resMapText'''
