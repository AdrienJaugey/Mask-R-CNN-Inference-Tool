"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import json
import logging
import random
import shutil
import urllib.request
import warnings
from distutils.version import LooseVersion

import cv2
import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform

from mrcnn.Config import Config
from mrcnn.visualize import create_multiclass_mask
from datasetTools import datasetDivider as dD

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Masks
############################################################

def reduce_memory(results, config: Config, allow_sparse=True):
    """
    Minimize all masks in the results dict from inference
    :param results: dict containing results of the inference
    :param config: the config object
    :param allow_sparse: if False, will only keep biggest region of a mask
    :return:
    """
    _masks = results['masks']
    _bbox = results['rois']
    if not allow_sparse:
        emptyMasks = []
        for idx in range(results['masks'].shape[-1]):
            mask = unsparse_mask(results['masks'][:, :, idx])
            if mask is None:
                emptyMasks.append(idx)
            else:
                results['masks'][:, :, idx] = mask
        if len(emptyMasks) > 0:
            results['scores'] = np.delete(results['scores'], emptyMasks)
            results['class_ids'] = np.delete(results['class_ids'], emptyMasks)
            results['masks'] = np.delete(results['masks'], emptyMasks, axis=2)
            results['rois'] = np.delete(results['rois'], emptyMasks, axis=0)
        results['rois'] = extract_bboxes(results['masks'])
    results['masks'] = minimize_mask(results['rois'], results['masks'], config.get_mini_mask_shape())
    return results


def get_mask_area(mask, verbose=0):
    """
    Computes mask area
    :param mask: the array representing the mask
    :param verbose: 0 : nothing, 1+ : errors/problems
    :return: the area of the mask and verbose output (None when nothing to print)
    """
    maskHistogram = dD.getBWCount(mask)
    display = None
    if verbose > 0:
        nbPx = mask.shape[0] * mask.shape[1]
        tempSum = maskHistogram[0] + maskHistogram[1]
        if tempSum != nbPx:
            display = "Histogram pixels {} != total pixels {}".format(tempSum, nbPx)
    return maskHistogram[1], display


def unsparse_mask(base_mask):
    """
    Return mask with only its biggest part
    :param base_mask: the mask image as np.bool or np.uint8
    :return: the main part of the mask as a same shape image and type
    """
    # http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
    # https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
    # Convert to np.uint8 if not before processing
    convert = False
    if type(base_mask[0, 0]) is np.bool_:
        convert = True
        base_mask = base_mask.astype(np.uint8) * 255
    # Padding the mask so that parts on edges will get correct area
    base_mask = np.pad(base_mask, 1, mode='constant', constant_values=0)
    res = np.zeros_like(base_mask, dtype=np.uint8)

    # Detecting contours and keeping only one with biggest area
    contours, _ = cv2.findContours(base_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        if len(contours) > 1:  # If only one region, reconstructing mask is useless
            biggest_part = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            # Drawing the biggest part on the result mask
            cv2.fillPoly(res, pts=[biggest_part], color=255)
        else:
            res = base_mask
        # Removing padding of the mask
        res = res[1:-1, 1:-1]
        return res.astype(np.bool) if convert else res
    else:
        return None


############################################################
#  Bounding Boxes
############################################################
def in_roi(roi_to_test, roi, epsilon=0):
    """
    Tests if the RoI to test is included in the given RoI
    :param roi_to_test: the RoI/bbox to test
    :param roi: the RoI that should include the one to test
    :param epsilon: margin of the RoI to allow boxes that are not exactly inside
    :return: True if roi_to_test is included in roi
    """
    res = True
    i = 0
    while i < 4 and res:
        res = res and (roi[i % 2] - epsilon <= roi_to_test[i] <= roi[i % 2 + 2] + epsilon)
        i += 1
    return res


def get_bbox_area(roi):
    """
    Returns the bbox area
    :param roi: the bbox to use
    :return: area of the given bbox
    """
    return (roi[3] - roi[1]) * (roi[2] - roi[0])


def get_bboxes_intersection(roiA, roiB):
    """
    Computes the intersection area of two bboxes
    :param roiA: the first bbox
    :param roiB: the second bbox
    :return: the area of the intersection
    """
    xInter = min(roiA[3], roiB[3]) - max(roiA[1], roiB[1])
    yInter = min(roiA[2], roiB[2]) - max(roiA[0], roiB[0])
    return max(xInter, 0) * max(yInter, 0)


def global_bbox(roiA, roiB):
    """
    Returns the bbox enclosing two given bboxes
    :param roiA: the first bbox
    :param roiB: the second bbox
    :return: the enclosing bbox
    """
    return np.array([min(roiA[0], roiB[0]), min(roiA[1], roiB[1]), max(roiA[2], roiB[2]), max(roiA[3], roiB[3])])


def shift_bbox(roi, customShift=None):
    """
    Shifts bbox coordinates so that min x and min y equal 0
    :param roi: the roi/bbox to transform
    :param customShift: custom x and y shift as (yShift, xShift)
    :return: the shifted bbox
    """
    yMin, xMin, yMax, xMax = roi
    if customShift is None:
        return np.array([0, 0, yMax - yMin, xMax - xMin])
    else:
        return np.array([max(yMin - customShift[0], 0), max(xMin - customShift[1], 0),
                         max(yMax - customShift[0], 0), max(xMax - customShift[1], 0)])


def expand_masks(mini_mask1, roi1, mini_mask2, roi2):
    """
    Expands two masks while keeping their relative position
    :param mini_mask1: the first mini mask
    :param roi1: the first mask bbox/roi
    :param mini_mask2: the second mini mask
    :param roi2: the second mask bbox/roi
    :return: mask1, mask2
    """
    roi1And2 = global_bbox(roi1, roi2)
    shifted_roi1And2 = shift_bbox(roi1And2)
    shifted_roi1 = shift_bbox(roi1, customShift=roi1And2[:2])
    shifted_roi2 = shift_bbox(roi2, customShift=roi1And2[:2])
    mask1 = expand_mask(shifted_roi1, mini_mask1, shifted_roi1And2[2:])
    mask2 = expand_mask(shifted_roi2, mini_mask2, shifted_roi1And2[2:])
    return mask1, mask2


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    soleMask = False
    if len(mask.shape) != 3:
        _mask = np.expand_dims(mask, 2)
        soleMask = True
    else:
        _mask = mask
    boxes = np.zeros([_mask.shape[-1], 4], dtype=np.int32)
    for i in range(_mask.shape[-1]):
        m = _mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2]).astype(np.int32)
    return boxes[0] if soleMask else boxes


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # TODO Possible improvements: using another structure to save overlaps as a lot of bboxes overlaps with only a few ?
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def compute_overlaps_masks(masks1, boxes1, masks2, boxes2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    res = np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return res

    matching_boxes = compute_overlaps(boxes1, boxes2)
    idx, idy = np.nonzero(matching_boxes)
    matching_boxes = set(zip(idx, idy))

    for idMask1, idMask2 in matching_boxes:
        mask1, mask2 = expand_masks(masks1[:, :, idMask1], boxes1[idMask1], masks2[:, :, idMask2], boxes2[idMask2])
        mask1Area, _ = get_mask_area(mask1)
        mask2Area, _ = get_mask_area(mask2)
        if mask1Area != 0 and mask2Area != 0:
            mask1AND2 = np.logical_and(mask1, mask2)
            intersection, _ = get_mask_area(mask1AND2)
            union = mask1Area + mask2Area - intersection
            res[idMask1, idMask2] = intersection / union
    return res


def non_max_suppression(boxes, scores, threshold):
    """
    Performs non-maximum suppression
    :param boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    :param scores: 1-D array of box scores.
    :param threshold: Float. IoU threshold to use for filtering.
    :return: indices of kept boxes
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indices of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


# TODO Delete
'''def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = compat.log(gt_height / height)
    dw = compat.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)'''


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    # TODO : Store mini masks as smaller float masks as in TF OD API
    soleMask = False
    if len(bbox.shape) != 2 and len(mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mask = np.expand_dims(mask, 2)
    else:
        _bbox = bbox
        _mask = mask
    mini_mask = np.zeros(mini_shape + (_mask.shape[-1],), dtype=bool)
    for i in range(_mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        # TODO Bool masks raising deprecation warning
        m = _mask[:, :, i].astype(bool).astype(np.uint8) * 255
        y1, x1, y2, x2 = _bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask[:, :, 0] if soleMask else mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    # TODO : Store mini masks as smaller float masks as in TF OD API
    if type(image_shape) is not tuple:
        image_shape = tuple(image_shape)
    soleMask = False
    if len(bbox.shape) != 2 and len(mini_mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mini_mask = np.expand_dims(mini_mask, 2)
    else:
        _bbox = bbox
        _mini_mask = mini_mask
    mask = np.zeros(image_shape[:2] + (_mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = _mini_mask[:, :, i].astype(bool).astype(np.uint8) * 255
        y1, x1, y2, x2 = _bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask[:, :, 0] if soleMask else mask


def minimize_mask_float(mask, bbox, output_shape=(28, 28), offset=32):
    """
    Minimizes given mask(s) to floating point masks of the given shape
    :param mask: mask as a 2-D uint8 ndarray of shape (H, W) or masks as a 3-D uint8 ndarray of shape (H, W, N)
    :param bbox: bbox as a 1-D uint8 ndarray of shape (4) or masks as a 2-D uint8 ndarray of shape (N, 4)
    :param output_shape: shape of the output mini-mask(s)
    :param offset: the offset on each side of the image part that will be resized (used to avoid
    :return: Minimized mask(s) in the same ndarray format as input ones but with output_shape as (H, W) and with float64
             dtype
    """
    soleMask = False
    if len(bbox.shape) != 2 and len(mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mask = np.expand_dims(mask, 2)
    else:
        _bbox = bbox
        _mask = mask
    mini_masks = np.zeros(output_shape + (_mask.shape[-1],), dtype=np.float64)
    for i in range(_mask.shape[-1]):
        # Computing mask shape with offset on all sides
        mask_shape = tuple(shift_bbox(_bbox[i][:4])[2:] + np.array([offset * 2] * 2))
        temp_mask = np.zeros(mask_shape, dtype=np.uint8)  # Empty mask
        y1, x1, y2, x2 = _bbox[i][:4]
        temp_mask[offset:-offset, offset:-offset] = _mask[y1:y2, x1:x2, i]  # Filling it with mask
        # Resizing to output shape
        mini_masks[:, :, i] = resize(temp_mask.astype(bool).astype(np.float64), output_shape)
    return mini_masks[:, :, 0] if soleMask else mini_masks


def expand_mask_float(mini_mask, bbox, output_shape=(1024, 1024), offset=32):
    """
    Expands given floating point mini-mask(s) back to binary mask(s) with the same shape as the image
    :param mini_mask: mini-mask as a 2-D uint8 ndarray of shape (H, W) or mini-masks as a 3-D uint8 ndarray of
                      shape (H, W, N)
    :param bbox: bbox as a 1-D uint8 ndarray of shape (4) or masks as a 2-D uint8 ndarray of shape (N, 4)
    :param output_shape:  shape of the output mask(s)
    :param offset: the offset on each side of the image part that will be resized (used to avoid
    :return: Expanded mask(s) in the same ndarray format as input ones but with output_shape as (H, W) and with uint8
             dtype
    """
    if type(output_shape) is not tuple:
        output_shape = tuple(output_shape)
    soleMask = False
    if len(bbox.shape) != 2 and len(mini_mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mini_mask = np.expand_dims(mini_mask, 2)
    else:
        _bbox = bbox
        _mini_mask = mini_mask
    masks = np.zeros(output_shape[:2] + (_mini_mask.shape[-1],), dtype=np.uint8)
    for i in range(_mini_mask.shape[-1]):
        mask_shape = tuple(shift_bbox(_bbox[i][:4])[2:] + np.array([offset * 2] * 2))
        resized_mask = resize(_mini_mask[:, :, i], mask_shape)
        y1, x1, y2, x2 = _bbox[i][:4]
        masks[y1:y2, x1:x2, i] = np.where(resized_mask[offset:-offset, offset:-offset] >= 0.5,
                                          255, 0).astype(np.uint8)
    return masks[:, :, 0] if soleMask else masks


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    # TODO : Float mini-mask seems to be the original format => refactor this to use instead of minimize_mask
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


# TODO Delete
'''############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)'''


############################################################
#  Miscellaneous
############################################################

def export_results(output_path: str, class_ids, boxes=None, masks=None, scores=None, bbox_areas=None, mask_areas=None):
    """
    Exports result dictionary to a JSON file for debug
    :param output_path: path to the output JSON file
    :param class_ids: value of the 'class_ids' key of results dictionary
    :param boxes: value of the 'class_ids' key of results dictionary
    :param masks: value of the 'class_ids' key of results dictionary
    :param scores: value of the 'class_ids' key of results dictionary
    :param bbox_areas: value of the 'bbox_areas' key of results dictionary
    :param mask_areas: value of the 'masks_areas' key of results dictionary
    :return: None
    """
    if type(class_ids) is dict:
        if 'rois' in class_ids:
            boxes = class_ids['rois']
        if 'masks' in class_ids:
            masks = class_ids['masks']
        if 'scores' in class_ids:
            scores = class_ids['scores']
        if 'bbox_areas' in class_ids:
            bbox_areas = class_ids['bbox_areas']
        if 'mask_areas' in class_ids:
            mask_areas = class_ids['mask_areas']
        class_ids = class_ids['class_ids']
    oneDArrays = [
        (class_ids, "class_ids", int),
        (scores, "scores", float),
        (bbox_areas, "bbox_areas", float),
        (mask_areas, "mask_areas", float),
    ]
    data = {key: [arrayType(v) for v in array] for array, key, arrayType in oneDArrays if array is not None}
    if boxes is not None:
        data["rois"] = [[int(v) for v in bbox] for bbox in boxes]
    if masks is not None:
        data["masks"] = [[[int(bool(v)) * 255 for v in row] for row in mask] for mask in masks]
    with open(output_path, 'w') as output:
        json.dump(data, output)


def import_results(input_path: str):
    """
    Imports result dictionary from JSON file for debug
    :param input_path: path to the input JSON file
    :return: results dictionary
    """
    with open(input_path, 'r') as inputFile:
        data = json.load(inputFile)
    keyType = {'rois': np.int32, 'masks': np.uint8, 'class_ids': int,
               'scores': float, 'bbox_areas': float, 'mask_areas': float}
    for key in data.keys():
        data[key] = np.array(data[key]).astype(keyType[key])
    return data


def classes_level(classes_hierarchy):
    """
    Return each level of the given class hierarchy with its classes
    :param classes_hierarchy: a structure made of list, int for classes of the same lvl, and dict to describe "key class
                              contains value class(es)". ex : [1, {2: [3, 4]}, {5: 6}] -> [[1, 2, 5], [3, 4, 6]]
    :return: list containing each classes of a level as a list : [[ lvl0 ], [ lvl1 ], ...]
    """
    if type(classes_hierarchy) is int:
        return [[classes_hierarchy]]  # Return a hierarchy with only one level containing the value
    elif type(classes_hierarchy) is list:
        res = []
        for element in classes_hierarchy:  # For each element of the list
            temp = classes_level(element)
            for lvl, indices in enumerate(temp):  # For each hierarchy level of the current element
                if len(indices) > 0:
                    if len(res) < lvl + 1:  # Adding a new level if needed
                        res.append([])
                    res[lvl].extend(indices)  # Fusing the current hierarchy level to list hierarchy one
        return res
    elif type(classes_hierarchy) is dict:
        res = [[]]
        for key in classes_hierarchy:
            res[0].append(key)  # Append key to lvl 0 classes
            if classes_hierarchy[key] is not None:
                temp = classes_level(classes_hierarchy[key])
                for lvl, indices in enumerate(temp):  # For each lvl of class inside the value of key element
                    if len(res) < lvl + 2:  # Adding a new level if needed
                        res.append([])
                    res[lvl + 1].extend(indices)  # Offsetting each level of the child to be relative to parent class
        return res


def remove_redundant_classes(classes_lvl, keepFirst=True):
    """
    Remove classes that appears more than once in the classes' levels
    :param classes_lvl: list of each level of classes as list : [[ lvl 0 ], [ lvl 1 ], ...]
    :param keepFirst: if True, class will be kept in the min level in which it is present, else in the max/last level.
    :return: [[ lvl 0 ], [ lvl 1 ], ...] with classes only appearing once
    """
    res = [[] for _ in classes_lvl]
    seenClass = []
    for lvlID, lvl in enumerate(classes_lvl[::1 if keepFirst else -1]):  # For each lvl in normal or reverse order
        for classID in lvl:
            if classID not in seenClass:  # Checking if the class ID has already been added or not
                seenClass.append(classID)  # Adding the class ID to the added ones
                res[lvlID if keepFirst else (-1 - lvlID)].append(classID)  # Adding the class to its level
    for lvl in res:  # Removing empty levels
        if len(lvl) == 0:
            res.remove(lvl)
    return res


def compute_confusion_matrix(image_shape: iter, expectedResults: dict, predictedResults: dict,
                             classes_hierarchy, num_classes: int, config: Config = None):
    """
    Computes confusion matrix at pixel precision
    :param image_shape: the initial image shape
    :param expectedResults: the expected results dict
    :param predictedResults: the predicted results dict
    :param classes_hierarchy: the classes hierarchy used to determine the order of the classes to apply masks
    :param num_classes: number of classes (max class ID)
    :param config: the config object of the AI
    :return: confusion matrix as a ndarray of shape (num_classes + 1, num_classes + 1), 0 being background class
    """
    expectedImg = create_multiclass_mask(image_shape, expectedResults, classes_hierarchy, config)
    predictedImg = create_multiclass_mask(image_shape, predictedResults, classes_hierarchy, config)
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            confusion_matrix[expectedImg[y, x]][predictedImg[y, x]] += 1
    return confusion_matrix


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


# TODO Delete
# def compute_matches(gt_boxes, gt_class_ids, gt_masks,
#                     pred_boxes, pred_class_ids, pred_scores, pred_masks,
#                     iou_threshold=0.5, score_threshold=0.0):
#     """Finds matches between prediction and ground truth instances.
#     Returns:
#         gt_match: 1-D array. For each GT box it has the index of the matched
#                   predicted box.
#         pred_match: 1-D array. For each predicted box, it has the index of
#                     the matched ground truth box.
#         overlaps: [pred_boxes, gt_boxes] IoU overlaps.
#     """
#     # Trim zero padding
#     # TODO: cleaner to do zero unpadding upstream
#     gt_boxes = trim_zeros(gt_boxes)
#     gt_masks = gt_masks[..., :gt_boxes.shape[0]]
#     pred_boxes = trim_zeros(pred_boxes)
#     pred_scores = pred_scores[:pred_boxes.shape[0]]
#     # Sort predictions by score from high to low
#     indices = np.argsort(pred_scores)[::-1]
#     pred_boxes = pred_boxes[indices]
#     pred_class_ids = pred_class_ids[indices]
#     pred_scores = pred_scores[indices]
#     pred_masks = pred_masks[..., indices]
#
#     # Compute IoU overlaps [pred_masks, gt_masks]
#     overlaps = compute_overlaps_masks(pred_masks, gt_masks)
#
#     # Loop through predictions and find matching ground truth boxes
#     match_count = 0
#     pred_match = -1 * np.ones([pred_boxes.shape[0]])
#     gt_match = -1 * np.ones([gt_boxes.shape[0]])
#     for i in range(len(pred_boxes)):
#         # Find best matching ground truth box
#         # 1. Sort matches by score
#         sorted_ixs = np.argsort(overlaps[i])[::-1]
#         # 2. Remove low scores
#         low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
#         if low_score_idx.size > 0:
#             sorted_ixs = sorted_ixs[:low_score_idx[0]]
#         # 3. Find the match
#         for j in sorted_ixs:
#             # If ground truth box is already matched, go to next one
#             if gt_match[j] > -1:
#                 continue
#             # If we reach IoU smaller than the threshold, end the loop (list is sorted so all the followings will be
#             # smaller too)
#             iou = overlaps[i, j]
#             if iou < iou_threshold:
#                 break
#             # Do we have a match?
#             if pred_class_ids[i] == gt_class_ids[j]:
#                 match_count += 1
#                 gt_match[j] = i
#                 pred_match[i] = j
#                 break
#
#     return gt_match, pred_match, overlaps


def compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes,
                    pred_class_ids, pred_scores, pred_masks,
                    ap_iou_threshold=0.5, min_iou_to_count=0.0,
                    nb_class=-1, confusion_iou_threshold=0.1,
                    classes_hierarchy=None, confusion_background_class=True, confusion_only_best_match=True):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    if nb_class > 0:
        bg = 1 if confusion_background_class else 0
        confusion_matrix = np.zeros((nb_class + bg, nb_class + bg), dtype=np.int64)
    else:
        confusion_matrix = None
        confusion_iou_threshold = 1.

    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, pred_boxes, gt_masks, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for pred_idx in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[pred_idx])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[pred_idx, sorted_ixs] < min_iou_to_count)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        match = False
        pred_class = pred_class_ids[pred_idx]
        for gt_idx in sorted_ixs:
            gt_class = gt_class_ids[gt_idx]
            # If classes_hierarchy is provided and (gt_class, pred_class) are parent/child classes we skip
            if classes_hierarchy is not None and \
                    ((gt_class in classes_hierarchy and pred_class in classes_hierarchy[gt_class]) or
                     (pred_class in classes_hierarchy and gt_class in classes_hierarchy[pred_class])):
                continue
            # If we reach IoU smaller than the threshold, end the loop (list is sorted so all the followings will be
            # smaller too)
            iou = overlaps[pred_idx, gt_idx]
            breakAP = iou < ap_iou_threshold
            breakConfusion = iou < confusion_iou_threshold
            if breakAP and breakConfusion:
                break

            if not breakConfusion and confusion_matrix is not None and (not confusion_only_best_match or not match):
                match = True
                if confusion_background_class:
                    confusion_matrix[gt_class][pred_class] += 1
                else:
                    confusion_matrix[gt_class - 1][pred_class - 1] += 1

            # If ground truth box is already matched, go to next one
            # TODO : Rework that part, specially for confusion matrix, we are counting positive predictions for each
            #        match with a gt_mask not only the first time
            if gt_match[gt_idx] > -1:
                continue

            if not breakAP:
                # Do we have a match?
                if pred_class == gt_class:
                    gt_match[gt_idx] = pred_idx
                    pred_match[pred_idx] = gt_idx
        # Something has been predicted but no ground truth annotation
        if confusion_matrix is not None and confusion_background_class and not match:
            confusion_matrix[0][pred_class] += 1
    # Looking for a ground truth box without overlapping prediction
    if confusion_matrix is not None and confusion_background_class:
        for gt_idx in range(len(gt_match)):
            if gt_match[gt_idx] == -1:
                if gt_class_ids[gt_idx] > nb_class:
                    print(f"Error : got class id = {gt_class_ids[gt_idx]} while max class id = {nb_class}")
                else:
                    confusion_matrix[gt_class_ids[gt_idx]][0] += 1
    return gt_match, pred_match, overlaps, confusion_matrix


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5, score_threshold=0.3,
               nb_class=-1, confusion_iou_threshold=0.3, classes_hierarchy=None,
               confusion_background_class=True, confusion_only_best_match=True):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps, confusion_matrix = compute_matches(
        gt_boxes=gt_boxes, gt_class_ids=gt_class_ids, gt_masks=gt_masks, min_iou_to_count=score_threshold,
        pred_boxes=pred_boxes, pred_class_ids=pred_class_ids, pred_masks=pred_masks, pred_scores=pred_scores,
        nb_class=nb_class, ap_iou_threshold=iou_threshold, confusion_iou_threshold=confusion_iou_threshold,
        classes_hierarchy=classes_hierarchy, confusion_background_class=confusion_background_class,
        confusion_only_best_match=confusion_only_best_match
    )
    if len(gt_class_ids) == len(pred_class_ids) == 0:
        return 1., 1., 1., overlaps, confusion_matrix
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    for i in range(len(recalls)):
        if np.isnan(recalls[i]):
            recalls[i] = 0.
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps, confusion_matrix


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps, _ = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print(f"AP @{iou_threshold:.2f}:\t {ap:.3f}")
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print(f"AP @{iou_thresholds[0]:.2f}-{iou_thresholds[-1]:.2f}:\t {AP:.3f}")
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# TODO delete
'''# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result'''


def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
