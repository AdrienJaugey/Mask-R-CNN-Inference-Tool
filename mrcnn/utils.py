"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import logging
import os
import random
from time import time, ctime
import cv2
import numpy as np
import tensorflow as tf
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion
from datasetTools import datasetDivider as div
from datasetTools.AnnotationAdapter import AnnotationAdapter
from datasetTools.ASAPAdapter import ASAPAdapter
from datasetTools.LabelMeAdapter import LabelMeAdapter
from common_utils import formatTime, progressBar, combination

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"


############################################################
#  Results Post-Processing
############################################################

def reduce_memory(results, config):
    _masks = results['masks']
    _bbox = results['rois']
    results['masks'] = minimize_mask(_bbox, _masks, config.MINI_MASK_SHAPE)
    return results


def fuse_results(results, image_shape, division_size=1024, min_overlap_part=0.33):
    """
    Fuse results of multiple predictions (divisions for example)
    :param results: list of the results of the predictions
    :param image_shape: the input image shape
    :param division_size: Size of a division
    :param min_overlap_part: Minimum overlap of divisions
    :return: same structure contained in results
    """
    # Getting base input image information
    div_side_length = results[0]['masks'].shape[0]
    use_mini_mask = division_size != "noDiv" and division_size != div_side_length
    height, width, _ = image_shape
    xStarts = [0] if division_size == "noDiv" else div.computeStartsOfInterval(width, division_size,
                                                                               min_overlap_part=min_overlap_part)
    yStarts = [0] if division_size == "noDiv" else div.computeStartsOfInterval(height, division_size,
                                                                               min_overlap_part=min_overlap_part)
    widthRatio = width / div_side_length
    heightRatio = height / div_side_length
    # Counting total sum of predicted masks
    size = 0
    for res in results:
        size += len(res['scores'])

    # Initialisation of arrays
    if use_mini_mask:
        masks = np.zeros((div_side_length, div_side_length, size), dtype=bool)
    else:
        masks = np.zeros((height, width, size), dtype=bool)
    scores = np.zeros(size)
    rois = np.zeros((size, 4), dtype=int)
    class_ids = np.zeros(size, dtype=int)

    # Iterating through divisions results
    lastIndex = 0
    for res in results:
        # Getting the division ID based on iterator or given ids and getting its coordinates
        divId = res["div_id"]
        xStart, xEnd, yStart, yEnd = div.getDivisionByID(xStarts, yStarts, divId, division_size)

        # Formatting and adding all the division's predictions to global ones
        for prediction_index in range(len(res['scores'])):
            scores[lastIndex] = res['scores'][prediction_index]
            class_ids[lastIndex] = res['class_ids'][prediction_index]

            if use_mini_mask:
                masks[:, :, lastIndex] = res['masks'][:, :, prediction_index]
            elif division_size != "noDiv":
                mask = res['masks'][:, :, prediction_index]
                if mask.shape[0] != division_size or mask.shape[1] != division_size:
                    tempWidthRatio = division_size / mask.shape[1]
                    tempHeightRatio = division_size / mask.shape[0]
                    roi = res['rois'][prediction_index]
                    roi[0] *= tempHeightRatio
                    roi[1] *= tempWidthRatio
                    roi[2] *= tempHeightRatio
                    roi[3] *= tempWidthRatio
                    mask = cv2.resize(np.uint8(mask), (division_size, division_size))
                masks[yStart:yEnd, xStart:xEnd, lastIndex] = mask
            else:
                mask = np.uint8(res['masks'][:, :, prediction_index])
                masks[:, :, lastIndex] = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)

            roi = res['rois'][prediction_index].copy()
            # y1, x1, y2, x2
            if division_size != "noDiv":
                roi[0] += yStart
                roi[1] += xStart
                roi[2] += yStart
                roi[3] += xStart
            else:
                roi[0] *= heightRatio
                roi[1] *= widthRatio
                roi[2] *= heightRatio
                roi[3] *= widthRatio
            rois[lastIndex] = roi

            lastIndex += 1

    # Formatting returned result
    fused_results = {
        "rois": rois,
        "class_ids": class_ids,
        "scores": scores,
        "masks": masks
    }
    return fused_results


def comparePriority(main_class_id, other_class_id, priority_table=None):
    """
    Compare priority of given class ids
    :param main_class_id: the main/current class id
    :param other_class_id: the other class id you want to compare to
    :param priority_table: the priority table to get the priority in
    :return: 1 if main has priority, -1 if other has priority, 0 if no one has priority or in case of parameter error
    """
    # Return 0 if no priority table given, if it has bad dimensions or a class_id is not in the correct range
    if priority_table is None:
        return 0
    elif not (len(priority_table) == len(priority_table[0])
              and 0 <= main_class_id < len(priority_table)
              and 0 <= other_class_id < len(priority_table)):
        return 0
    if priority_table[main_class_id][other_class_id]:
        return 1
    elif priority_table[other_class_id][main_class_id]:
        return -1
    else:
        return 0


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


def get_mask_area(mask, verbose=0):
    """
    Computes mask area
    :param mask: the array representing the mask
    :param verbose: 0 : nothing, 1+ : errors/problems
    :return: the area of the mask and verbose output (None when nothing to print)
    """
    maskHistogram = div.getBWCount(mask)
    display = None
    if verbose > 0:
        nbPx = mask.shape[0] * mask.shape[1]
        tempSum = maskHistogram[0] + maskHistogram[1]
        if tempSum != nbPx:
            display = "Histogram pixels {} != total pixels {}".format(tempSum, nbPx)
    return maskHistogram[1], display


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


def fuse_masks(fused_results, bb_threshold=0.1, mask_threshold=0.1, config=None, displayProgress: str = None,
               verbose=0):
    """
    Fuses overlapping masks of the same class
    :param fused_results: the fused predictions results
    :param bb_threshold: least part of bounding boxes overlapping to continue checking
    :param mask_threshold: idem but with mask
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return: fused_results with only fused masks
    """
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']

    bbAreas = np.ones(len(class_ids), dtype=int) * -1
    maskAreas = np.ones(len(class_ids), dtype=int) * -1
    fusedWith = np.ones(len(class_ids), dtype=int) * -1
    maskCount = np.ones(len(class_ids), dtype=int)
    toDelete = []
    if displayProgress is not None:
        total = combination(len(class_ids), 2)
        current = 1
        start_time = time()
        duration = ""
    for idxI, roi1 in enumerate(rois):
        # Computation of the bounding box area if not done yet
        if bbAreas[idxI] == -1:
            bbAreas[idxI] = get_bbox_area(roi1)

        for idxJ in range(idxI + 1, len(rois)):
            if displayProgress is not None:
                if current == total:
                    duration = f" Duration = {formatTime(time() - start_time)}"
                if current % 20 == 0 or current == total:
                    progressBar(current, total, prefix=displayProgress, suffix=duration)
                current += 1
            '''###################################
            ###     CHECKING BBOX OVERLAP      ###
            ###################################'''
            # If masks are not from the same class or have been fused with same mask, we skip them
            if idxI == idxJ or class_ids[idxI] != class_ids[idxJ] \
                    or (fusedWith[idxI] == fusedWith[idxJ] and fusedWith[idxI] != -1):
                continue

            hadPrinted = False
            roi2 = rois[idxJ]

            # Computation of the bounding box area if not done yet
            if bbAreas[idxJ] == -1:
                bbAreas[idxJ] = get_bbox_area(roi2)

            # Computation of the bb intersection
            bbIntersection = get_bboxes_intersection(roi1, roi2)

            # We skip next part if bb intersection not representative enough
            partOfRoI1 = bbIntersection / bbAreas[idxI]
            partOfRoI2 = bbIntersection / bbAreas[idxJ]

            '''###################################
            ###     CHECKING MASK OVERLAP      ###
            ###################################'''
            if partOfRoI1 > bb_threshold or partOfRoI2 > bb_threshold:
                if verbose > 1:
                    hadPrinted = True
                    print("[{:03d}/{:03d}] Enough RoI overlap".format(idxI, idxJ))

                mask1 = masks[:, :, idxI]
                mask2 = masks[:, :, idxJ]

                if config is not None and config.USE_MINI_MASK:
                    mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                if maskAreas[idxI] == -1:
                    maskAreas[idxI], verbose_output = get_mask_area(mask1, verbose=verbose)
                    if verbose_output is not None:
                        hadPrinted = True
                        print("[{:03d}] {}".format(idxI, verbose_output))

                if maskAreas[idxJ] == -1:
                    maskAreas[idxJ], verbose_output = get_mask_area(mask2, verbose=verbose)
                    if verbose_output is not None:
                        hadPrinted = True
                        print("[{:03d}] {}".format(idxJ, verbose_output))

                # Computing intersection of mask 1 and 2 and computing its area
                mask1AND2 = np.logical_and(mask1, mask2)
                mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                partOfMask1 = mask1AND2Area / maskAreas[idxI]
                partOfMask2 = mask1AND2Area / maskAreas[idxJ]

                if verbose > 0:
                    verbose_output = "[{:03d}] Intersection representing more than 100% of the mask : {:3.2f}%"
                    if not (0 <= partOfMask1 <= 1):
                        hadPrinted = True
                        print(verbose_output.format(idxI, partOfMask1 * 100))

                    if not (0 <= partOfMask2 <= 1):
                        hadPrinted = True
                        print(verbose_output.format(idxJ, partOfMask2 * 100))

                    if verbose > 1:
                        print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask1 * 100, idxI))
                        print("[OR] {:5.2f}% of mask [{:03d}]".format(partOfMask2 * 100, idxJ))

                '''####################
                ###     FUSION      ###
                ####################'''
                if partOfMask1 > mask_threshold or partOfMask2 > mask_threshold:
                    # If the first mask has already been fused with another mask, we will fuse with the "parent" one
                    if fusedWith[idxI] == fusedWith[idxJ] == -1:  # No mask fused
                        receiver = idxI
                        giver = idxJ
                    elif fusedWith[idxI] != -1:  # I fused
                        if fusedWith[idxJ] == -1:  # I fused, not J
                            receiver = fusedWith[idxI]
                            giver = idxJ
                        else:  # I and J fused but not with each other (previous test)
                            receiver = min(fusedWith[idxI], fusedWith[idxJ])
                            giver = max(fusedWith[idxI], fusedWith[idxJ])
                            for idx in range(len(fusedWith)):  # As giver will be deleted, we have to update the list
                                if fusedWith[idx] == giver:
                                    fusedWith[idx] = receiver
                    else:  # J fused, not I (previous test)
                        receiver = fusedWith[idxJ]
                        giver = idxI

                    fusedWith[giver] = receiver
                    toDelete.append(giver)

                    if verbose > 1:
                        print("[{:03d}] Fusion with [{:03d}]".format(giver, receiver))

                    receiverMask = masks[:, :, receiver]
                    giverMask = masks[:, :, giver]
                    if config is not None and config.USE_MINI_MASK:
                        receiverRoI = rois[receiver]
                        giverRoI = rois[giver]
                        receiverMask, giverMask = expand_masks(receiverMask, receiverRoI, giverMask, giverRoI)
                    fusedMask = np.logical_or(receiverMask, giverMask)

                    if verbose > 1:
                        verbose_output = "[{idReceiver:03d}] Receiver Mask {when} fusion:\n"
                        verbose_output += "\tROI = {roi}\n"
                        verbose_output += "\tROI area = {roiArea}\n"
                        verbose_output += "\tMask area = {maskArea}\n"
                        verbose_output += "\tScore = {score}\n"
                        verbose_output += "\tMask count = {count}\n"
                        print(verbose_output.format(when="before", idReceiver=receiver, roi=rois[receiver],
                                                    roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                    score=scores[receiver], count=maskCount[receiver]))

                    # Updating the receiver mask's infos
                    rois[receiver] = global_bbox(rois[receiver], rois[giver])
                    if config is not None and config.USE_MINI_MASK:
                        shifted_fusedRoI = shift_bbox(rois[receiver])
                        masks[:, :, receiver] = minimize_mask(shifted_fusedRoI, fusedMask, config.MINI_MASK_SHAPE)
                    else:
                        masks[:, :, receiver] = fusedMask
                    bbAreas[receiver] = get_bbox_area(rois[receiver])
                    maskAreas[receiver], _ = get_mask_area(fusedMask)
                    scores[receiver] = (scores[receiver] * maskCount[receiver] + scores[idxJ])
                    maskCount[receiver] += 1
                    scores[receiver] /= maskCount[receiver]

                    if verbose > 1:
                        print(verbose_output.format(when="after", idReceiver=receiver, roi=rois[receiver],
                                                    roiArea=bbAreas[receiver], maskArea=maskAreas[receiver],
                                                    score=scores[receiver], count=maskCount[receiver]))

                if verbose > 0 and hadPrinted:
                    print(flush=True)
    if displayProgress is not None and current <= total:
        duration = f" Duration = {formatTime(time() - start_time)}"
        progressBar(total, total, prefix=displayProgress, suffix=duration, forceNewLine=True)
    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def filter_fused_masks(fused_results, bb_threshold=0.5, mask_threshold=0.9, priority_table=None,
                       config=None, displayProgress: str = None, verbose=0):
    """
    Post-prediction filtering to remove non-sense predictions
    :param fused_results: the results after fusion
    :param bb_threshold: the least part of overlapping bounding boxes to continue checking
    :param mask_threshold: the least part of a mask contained in another for it to be deleted
    :param priority_table: the priority table used to compare classes
    :param config: the config to get mini_mask informations
    :param displayProgress: if string given, prints a progress bar using this as prefix
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return:
    """
    rois = fused_results['rois']
    masks = fused_results['masks']
    scores = fused_results['scores']
    class_ids = fused_results['class_ids']
    bbAreas = fused_results['bbox_areas']
    maskAreas = fused_results['mask_areas']
    toDelete = []
    if displayProgress is not None:
        total = combination(len(class_ids), 2)
        current = 1
        start_time = time()
        duration = ""
    for i, roi1 in enumerate(rois):
        # If this RoI has already been selected for deletion, we skip it
        if i in toDelete:
            continue

        # If the area of this RoI has not been computed
        if bbAreas[i] == -1:
            bbAreas[i] = get_bbox_area(roi1)

        # Then we check for each RoI that has not already been checked
        for j in range(i + 1, len(rois)):
            if displayProgress is not None:
                if current == total:
                    duration = f" Duration = {formatTime(time() - start_time)}"
                if current % 20 == 0 or current == total:
                    progressBar(current, total, prefix=displayProgress, suffix=duration)
                current += 1
            if j in toDelete:
                continue
            roi2 = rois[j]

            # We want only one prediction class to be vessel
            priority = comparePriority(class_ids[i] - 1, class_ids[j] - 1, priority_table)
            if priority == 0:
                continue

            # If the area of the 2nd RoI has not been computed
            if bbAreas[j] == -1:
                bbAreas[j] = get_bbox_area(roi2)

            # Computation of the bb intersection
            intersection = get_bboxes_intersection(roi1, roi2)

            # We skip next part if bb intersection not representative enough
            partOfR1 = intersection / bbAreas[i]
            partOfR2 = intersection / bbAreas[j]
            if partOfR1 > bb_threshold or partOfR2 > bb_threshold:
                # Getting first mask and computing its area if not done yet
                mask1 = masks[:, :, i]
                mask2 = masks[:, :, j]

                if config is not None and config.USE_MINI_MASK:
                    mask1, mask2 = expand_masks(mask1, roi1, mask2, roi2)

                if maskAreas[i] == -1:
                    maskAreas[i], _ = get_mask_area(mask1, verbose=verbose)
                    if maskAreas[i] == 0:
                        print(i, maskAreas[i])

                # Getting second mask and computing its area if not done yet
                if maskAreas[j] == -1:
                    maskAreas[j], _ = get_mask_area(mask2, verbose=verbose)
                    if maskAreas[j] == 0:
                        print(j, maskAreas[j])

                # Computing intersection of mask 1 and 2 and computing its area
                mask1AND2 = np.logical_and(mask1, mask2)
                mask1AND2Area, _ = get_mask_area(mask1AND2, verbose=verbose)
                partOfMask1 = mask1AND2Area / maskAreas[i]
                partOfMask2 = mask1AND2Area / maskAreas[j]

                # We check if the common area represents more than the vessel_threshold of the non-vessel mask
                if priority == -1 and partOfMask1 > mask_threshold:
                    if verbose > 0:
                        print("[{:03d}/{:03d}] Kept class = {}\tRemoved Class = {}".format(i, j, class_ids[i],
                                                                                           class_ids[j]))
                    toDelete.append(i)
                elif priority == 1 and partOfMask2 > mask_threshold:
                    if verbose > 0:
                        print("[{:03d}/{:03d}] Kept class = {}\tRemoved Class = {}".format(i, j, class_ids[i],
                                                                                           class_ids[j]))
                    toDelete.append(j)
    if displayProgress is not None and current <= total:
        duration = f" Duration = {formatTime(time() - start_time)}"
        progressBar(total, total, prefix=displayProgress, suffix=duration, forceNewLine=True)
    # Deletion of unwanted results
    scores = np.delete(scores, toDelete)
    class_ids = np.delete(class_ids, toDelete)
    bbAreas = np.delete(bbAreas, toDelete)
    maskAreas = np.delete(maskAreas, toDelete)
    masks = np.delete(masks, toDelete, axis=2)
    rois = np.delete(rois, toDelete, axis=0)
    return {"rois": rois, "bbox_areas": bbAreas, "class_ids": class_ids,
            "scores": scores, "masks": masks, "mask_areas": maskAreas}


def getCountAndArea(results: dict, classesInfo: dict, selectedClasses: [str], config=None):
    """
    Computing count and area of classes from results
    :param results: the results
    :param classesInfo: the dict with information about name, inference id, id... parameters of each class
    :param selectedClasses: list of classes' names that you want to get statistics on
    :return: Dict of "className": {"count": int, "area": int} elements for each classes
    """
    res = {}

    rois = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']

    # Getting the inferenceIDs of the wanted classes
    selectedClassesID = {}
    for classInfo in classesInfo:
        if classInfo["name"] in selectedClasses:
            selectedClassesID[classInfo["inferenceID"]] = classInfo["name"]
            res[classInfo["name"]] = {"count": 0, "area": 0}

    # For each predictions, if class ID matching with one we want
    for index, classID in enumerate(class_ids):
        if classID in selectedClassesID.keys():
            # Getting current values of count and area
            className = selectedClassesID[classID]
            res[className]["count"] += 1
            # Getting the area of current mask
            if config is not None and config.USE_MINI_MASK:
                shifted_roi = shift_bbox(rois[index])
                mask = expand_mask(shifted_roi, masks[:, :, index], shifted_roi[2:])
            else:
                yStart, xStart, yEnd, xEnd = rois[index]
                mask = masks[yStart:yEnd, xStart:xEnd, index]
            mask = mask.astype(np.uint8)
            if "mask_areas" in results and results['mask_areas'][index] != -1:
                area = int(results['mask_areas'][index])
            else:
                area, _ = get_mask_area(mask)
            res[className]["area"] += area  # Cast to int to avoid "json 'int64' not serializable"

    return res


def getPoints(mask, xOffset=0, yOffset=0, epsilon=1,
              show=False, waitSeconds=10, info=False):
    """
    Return a list of points describing the given mask as a polygon
    :param mask: the mask you want the points
    :param xOffset: if using a RoI the x-axis offset used
    :param yOffset: if using a RoI the y-axis offset used
    :param epsilon: epsilon parameter of cv2.approxPolyDP() method
    :param show: whether you want or not to display the approximated mask so you can see it
    :param waitSeconds: time in seconds to wait before closing automatically the displayed masks, or press ESC to close
    :param info: whether you want to display some information (mask size, number of predicted points, number of
    approximated points...) or not
    :return: 2D-array of points coordinates : [[x, y]]
    """
    pts = None
    contours, _ = cv2.findContours(mask, method=cv2.RETR_TREE, mode=cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # https://stackoverflow.com/questions/41879315/opencv-visualize-polygonal-curves-extracted-with-cv2-approxpolydp
        # Finding biggest area
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        res = cv2.approxPolyDP(cnt, epsilon, True)
        pts = []
        for point in res:
            # Casting coordinates to int, not doing this makes crash json dump
            pts.append([int(point[0][0] + xOffset), int(point[0][1] + yOffset)])

        if info:
            maskHeight, maskWidth = mask.shape
            nbPtPred = contours[0].shape[0]
            nbPtApprox = len(pts)
            print("Mask size : {}x{}".format(maskWidth, maskHeight))
            print("Nb points prediction : {}".format(nbPtPred))
            print("Nb points approx : {}".format(nbPtApprox))
            print("Compression rate : {:5.2f}%".format(nbPtPred / nbPtApprox * 100))
            temp = np.array(pts)
            xMin = np.amin(temp[:, 0])
            xMax = np.amax(temp[:, 0])
            yMin = np.amin(temp[:, 1])
            yMax = np.amax(temp[:, 1])
            print("{} <= X <= {}".format(xMin, xMax))
            print("{} <= Y <= {}".format(yMin, yMax))
            print()

        if show:
            img = np.zeros(mask.shape, np.int8)
            img = cv2.drawContours(img, [res], -1, 255, 2)
            cv2.imshow('before {}'.format(img.shape), mask * 255)
            cv2.imshow("approxPoly", img * 255)
            cv2.waitKey(max(waitSeconds, 1) * 1000)

    return pts


def export_annotations(image_info: dict, results: dict, classes_info: [{int, str, str}],
                       adapter: AnnotationAdapter.__class__, save_path="predicted", config=None, verbose=0):
    """
    Exports predicted results to an XML annotation file using given XMLExporter
    :param image_info: Dict with at least {"NAME": str, "HEIGHT": int, "WIDTH": int} about the inferred image
    :param results: inference results of the image
    :param classes_info: list of class names, including background
    :param adapter: class inheriting XMLExporter
    :param save_path: path to the dir you want to save the annotation file
    :param config: the config to get mini_mask informations
    :param verbose: verbose level of the method (0 = nothing, 1 = information)
    :return: None
    """
    isASAPAdapter = adapter is ASAPAdapter
    isLabelMeAdapter = adapter is LabelMeAdapter
    assert not (isASAPAdapter and isLabelMeAdapter)

    if verbose > 0:
        if isASAPAdapter:
            print("Exporting to ASAP annotation file format.")
        if isLabelMeAdapter:
            print("Exporting to LabelMe annotation file format.")

    rois = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    height = masks.shape[0]
    width = masks.shape[1]
    xmlData = adapter({"name": image_info['NAME'], "height": image_info['HEIGHT'],
                       'width': image_info['WIDTH'], 'format': image_info['IMAGE_FORMAT']},
                      verbose=verbose)
    # For each prediction
    for i in range(masks.shape[2]):
        if config is not None and config.USE_MINI_MASK:
            shifted_roi = shift_bbox(rois[i])
            shifted_roi += [5, 5, 5, 5]
            image_size = shifted_roi[2:] + [5, 5]
            mask = expand_mask(shifted_roi, masks[:, :, i], image_size)
            yStart, xStart = rois[i][:2] - [5, 5]
        else:
            # Getting the RoI coordinates and the corresponding area
            # y1, x1, y2, x2
            yStart, xStart, yEnd, xEnd = rois[i]
            yStart = max(yStart - 10, 0)
            xStart = max(xStart - 10, 0)
            yEnd = min(yEnd + 10, height)
            xEnd = min(xEnd + 10, width)
            mask = masks[yStart:yEnd, xStart:xEnd, i]

        # Getting list of points coordinates and adding the prediction to XML
        points = getPoints(np.uint8(mask), xOffset=xStart, yOffset=yStart, show=False, waitSeconds=0, info=False)
        if points is None:
            continue
        classInfo = None
        iterator = 0
        # Find the first class not to be ignored with the same id
        while classInfo is None:
            temp = classes_info[iterator]
            if not temp["ignore"] and temp["inferenceID"] == class_ids[i]:
                classInfo = temp
            iterator += 1
        xmlData.addAnnotation(classInfo, points)

    for classInfo in classes_info:
        if classInfo["id"] == 0:
            continue
        xmlData.addAnnotationClass(classInfo)

    os.makedirs(save_path, exist_ok=True)
    xmlData.saveToFile(save_path, image_info['NAME'])


############################################################
#  Bounding Boxes
############################################################

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


def compute_overlaps_masks(masks1, boxes1, masks2, boxes2, config=None):
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
        mask1AND2 = np.logical_and(mask1, mask2)
        intersection, _ = get_mask_area(mask1AND2)
        union = mask1Area + mask2Area - intersection
        res[idMask1, idMask2] = intersection / union
    '''
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    '''
    return res


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
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

    # Get indicies of boxes sorted by scores (highest first)
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


def apply_box_deltas(boxes, deltas):
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
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

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

    return np.stack([dy, dx, dh, dw], axis=1)


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
        m = _mask[:, :, i].astype(bool)
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
        m = _mini_mask[:, :, i]
        y1, x1, y2, x2 = _bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask[:, :, 0] if soleMask else mask


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
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


############################################################
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
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def compute_matches(gt_boxes, gt_class_ids, gt_masks, pred_boxes,
                    pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0, config=None,
                    nb_class=-1, confusion_iou_threshold=0.1):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream

    if nb_class > 0:
        confusion_matrix = np.zeros((nb_class + 1, nb_class + 1), dtype=np.int32)
    else:
        confusion_matrix = None
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
    overlaps = compute_overlaps_masks(pred_masks, pred_boxes, gt_masks, gt_boxes, config=config)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        nothing = True
        done = False
        for j in sorted_ixs:
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if confusion_matrix is not None and iou >= confusion_iou_threshold:
                nothing = False
                confusion_matrix[gt_class_ids[j]][pred_class_ids[i]] += 1

            # If ground truth box is already matched, go to next one
            # TODO : Rework that part, specially for confusion matrix, we are counting positive predictions for each
            #        match with a gt_mask not only the first time
            if not done and gt_match[j] > -1:
                continue

            if iou >= iou_threshold:
                # Do we have a match?
                if pred_class_ids[i] == gt_class_ids[j]:
                    match_count += 1
                    gt_match[j] = i
                    pred_match[i] = j
                    done = True
        # Something has been predicted but no ground truth annotation
        if confusion_matrix is not None and nothing:
            confusion_matrix[0][pred_class_ids[i]] += 1
    # Looking for a ground truth box without overlapping prediction
    if confusion_matrix is not None:
        for j in range(len(gt_match)):
            if gt_match[j] == -1:
                if gt_class_ids[j] > nb_class:
                    print("Error : got class id = {} while max class id = {}".format(gt_class_ids[j], nb_class))
                else:
                    confusion_matrix[gt_class_ids[j]][0] += 1
    return gt_match, pred_match, overlaps, confusion_matrix


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5, config=None,
               nb_class=-1, confusion_iou_threshold=0.1):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps, confusion_matrix = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold, config=config,
        nb_class=nb_class, confusion_iou_threshold=confusion_iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

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
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
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


# ## Batch Slicing
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

    return result


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
