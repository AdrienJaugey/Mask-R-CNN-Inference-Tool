import os
from enum import Enum
import json
import numpy as np

from mrcnn import utils
from datasetTools import datasetDivider as dD
from mrcnn.Config import Config, DynamicMethod


def get_count_and_area(results: dict, image_info: dict, selected_classes: [str], save=None,
                       display=True, config: Config = None, verbose=0):
    """
    Computing count and area of classes from results
    :param results: the results
    :param image_info: Dict containing informations about the image
    :param selected_classes: list of classes' names that you want to get statistics on
    :param save: if given, path to the json file that will contains statistics
    :param display: if True, will print the statistics
    :param config: the config to get mini_mask informations
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information, ...
    :return: Dict of "className": {"count": int, "area": int} elements for each classes
    """
    if config is None or (save is None and not display):
        return

    print(" - Computing statistics on predictions")

    rois = results['rois']
    masks = results['masks']
    class_ids = results['class_ids']
    indices = np.arange(len(class_ids))
    mini_mask_used = config.is_using_mini_mask()

    # Getting the inferenceIDs of the wanted classes
    if selected_classes == "all":
        selectedClassesID = {aClass['id']: aClass['name'] for aClass in config.get_classes_info()}
    else:
        selectedClassesID = {config.get_class_id(name): name for name in selected_classes}
        indices = indices[np.isin(class_ids, list(selectedClassesID.keys()))]
    res = {c: {"count": 0, "area": 0} for c in selectedClassesID.values()}

    # For each predictions, if class ID matching with one we want
    for index in indices:
        # Getting current values of count and area
        className = selectedClassesID[class_ids[index]]
        res[className]["count"] += 1
        # Getting the area of current mask
        if mini_mask_used:
            shifted_roi = utils.shift_bbox(rois[index])
            mask = utils.expand_mask(shifted_roi, masks[:, :, index], shifted_roi[2:])
        else:
            yStart, xStart, yEnd, xEnd = rois[index]
            mask = masks[yStart:yEnd, xStart:xEnd, index]
        mask = mask.astype(np.uint8)
        if "mask_areas" in results and results['mask_areas'][index] != -1:
            area = int(results['mask_areas'][index])
        else:
            area, _ = utils.get_mask_area(mask)
        res[className]["area"] += area  # Cast to int to avoid "json 'int64' not serializable"

    if 'BASE_CLASS' in image_info:
        res[image_info['BASE_CLASS']] = {"count": image_info['BASE_COUNT'], "area": image_info["BASE_AREA"]}
    if save is not None:
        with open(os.path.join(save, f"{image_info['NAME']}_stats.json"), "w") as saveFile:
            try:
                json.dump(res, saveFile, indent='\t')
            except TypeError:
                if verbose > 0:
                    print("    Failed to save statistics", flush=True)
    if display:
        for className in res:
            stat = res[className]
            print(f"    - {className} : count = {stat['count']}, area = {stat['area']} px")

    return res


def __get_count_and_area__(results, args: dict, config=None, display=True, verbose=0, dynargs=None):
    return get_count_and_area(results=results, image_info=dynargs['image_info'],
                              selected_classes=args.get('selected_classes', 'all'), save=dynargs.get('save', None),
                              display=display, config=config, verbose=verbose)


def mask_histo_per_base_mask(base_results, results, image_info, classes=None, box_epsilon: int = 0,
                             test_masks=True, mask_threshold=0.9, config: Config = None,
                             display_per_base_mask=False, display_global=False, save=None, verbose=0):
    """
    Return an histogram of the number of each mask of a class inside each base mask
    :param base_results: results of the previous inference mode or ground-truth
    :param results: results of the current inference mode or ground-truth
    :param image_info: Dict containing informations about the image
    :param classes: dict that link previous classes to current classes that we want to count
    :param box_epsilon: margin of the RoI to allow boxes that are not exactly inside
    :param test_masks: if True, will test that masks are at least 'mask_threshold' inside the base mask
    :param mask_threshold: threshold that will define if a mask is included inside the base mask
    :param config: Config object of the Inference Tool
    :param display_per_base_mask: if True, will display each base mask histogram
    :param display_global: if True, will display global histogram
    :param save: if given, will be used as directory path to save json file of
    :param verbose: 0 : nothing, 1+ : errors/problems, 2 : general information
    :return: global histogram of how many base masks contain a certain amount an included class mask
    """
    # If classes is None or empty, skip method
    if classes is None or classes == {} or config is None \
            or (save is None and not (display_per_base_mask or display_global)):
        return results

    print(" - Computing base masks histograms")

    if box_epsilon < 0:
        raise ValueError(f"box_epsilon ({box_epsilon}) cannot be negative")

    # Getting all the base data
    b_class_ids = base_results['class_ids']
    b_rois = base_results['rois']
    b_masks = base_results['masks']
    if 'histos' not in base_results:  # If histo does not exists, initiate it
        base_results['histos'] = np.empty(len(b_class_ids), dtype=object)
    histo = base_results['histos']
    b_indices = np.arange(len(b_class_ids), dtype=int)

    # Getting all the results/current data
    c_class_ids = results['class_ids']
    c_rois = results['rois']
    c_masks = results['masks']
    c_indices = np.arange(len(results['class_ids']), dtype=int)
    c_areas = results.get('mask_areas', np.ones(len(c_class_ids), dtype=int) * -1)

    # For each base class that we want to get an histogram of the included current classes
    for b_class in classes:
        # Retrieving indices of the base and associated current classes
        b_class_id = config.get_class_id(b_class, "previous")
        b_cur_idx = b_indices[np.isin(b_class_ids, [b_class_id])]
        if classes[b_class] == "all":
            c_cur_idx = c_indices
        else:
            c_class_id = [config.get_class_id(aClass) for aClass in classes[b_class]]
            c_cur_idx = c_indices[np.isin(c_class_ids, c_class_id)]
        for b_idx in b_cur_idx:  # For each base class mask
            b_roi = b_rois[b_idx]
            custom_shift = b_roi[:2] - box_epsilon
            padded_size = b_roi[2:] - b_roi[:2] + (box_epsilon * 2)
            if test_masks:
                b_mask = b_masks[..., b_idx]
                if config.is_using_mini_mask(config.get_previous_mode()):
                    b_shifted_roi = utils.shift_bbox(b_roi, custom_shift)
                    b_mask = utils.expand_mask(b_shifted_roi, b_mask, padded_size)
                else:
                    b_mask = np.pad(b_mask[b_roi[0]:b_roi[2], b_roi[1]:b_roi[3]], box_epsilon)
            if histo[b_idx] is None:
                histo[b_idx] = {}

            for c_idx in c_cur_idx:  # For each mask of one of the current classes
                c_roi = c_rois[c_idx]

                # If the bbox of the current mask is inside the base bbox
                if utils.in_roi(c_roi, b_roi, epsilon=box_epsilon):

                    if test_masks:  # If we have to check that masks are included
                        c_mask = c_masks[..., c_idx]
                        if config.is_using_mini_mask():
                            c_shifted_roi = utils.shift_bbox(c_roi, custom_shift)
                            c_mask = utils.expand_mask(c_shifted_roi, c_mask, padded_size)
                        else:
                            c_mask = np.pad(c_mask[b_roi[0]:b_roi[2], b_roi[1]:b_roi[3]], box_epsilon)
                        if c_areas[c_idx] == -1:
                            c_areas[c_idx] = dD.getBWCount(c_mask)[1]
                        c_mask = np.bitwise_and(b_mask, c_mask)
                        c_area_in = dD.getBWCount(c_mask)[1]
                        if c_area_in <= c_areas[c_idx] * mask_threshold:  # If the included part is not enough, skip it
                            continue
                    c_class = c_class_ids[c_idx]
                    if c_class not in histo[b_idx]:
                        histo[b_idx][c_class] = 0
                    histo[b_idx][c_class] += 1
    if display_per_base_mask:
        for idx, histogram in enumerate(base_results['histos']):
            if histogram is not None:
                print(f"    - mask n°{idx}:", ", ".join([f"{nb} {config.get_class_name(c, display=True)}"
                                                         for c, nb in histogram.items()]))
    global_histo = mask_to_class_histogram(base_results, config=config)
    if display_global:
        for class_, histogram in global_histo.items():
            print(f"    - {class_}:", ", ".join([f"{nb_elt} [{nb_mask} base mask{'s' if nb_mask > 1 else ''}]"
                                                 for nb_elt, nb_mask in histogram.items()]))
    if save is not None:
        temp = {'_comment': "<class A>: {N: <nb BASE masks with N class A masks>}"}
        temp.update(global_histo)
        with open(os.path.join(save, f'{image_info["NAME"]}_histo.json'), 'w') as saveFile:
            json.dump(temp, saveFile, indent='\t')
    return global_histo


def __mask_histogram_per_base_mask__(results, args: dict, config=None, display=True, verbose=0, dynargs=None):
    return mask_histo_per_base_mask(
        base_results=dynargs['base_res'], results=results, image_info=dynargs['image_info'],
        classes=args['classes'], box_epsilon=args.get('box_epsilon', 0), test_masks=args.get('test_masks', True),
        mask_threshold=args.get('mask_threshold', 0.9), config=config,
        display_per_base_mask=args.get('display_per_base_mask', False) and display, verbose=verbose,
        display_global=args.get('display_global', True) and display, save=dynargs.get('save', None)
    )


def mask_to_class_histogram(results, config: Config = None):
    """
    Gather all histograms into a general one that looses the information of 'which base mask contains which masks'
    :param results: the results containing per mask histograms
    :param config: the config (json as a dict)
    :return: the global histogram
    """
    if config is None:
        return
    noms = ['']
    for aClass in config.get_classes_info():
        noms.append(aClass['display_name'])
    histogram = {c: {} for c in noms[1:]}
    for mask_histogram in results['histos']:
        if mask_histogram is None:
            continue
        else:
            for eltClass in histogram:
                class_id = noms.index(eltClass)
                if class_id not in mask_histogram:
                    if 0 not in histogram[eltClass]:
                        histogram[eltClass][0] = 0
                    histogram[eltClass][0] += 1
                else:
                    nb = mask_histogram[class_id]
                    if nb not in histogram[eltClass]:
                        histogram[eltClass][nb] = 0
                    histogram[eltClass][nb] += 1
    return histogram


class StatisticsMethod(DynamicMethod):
    GET_COUNT_AND_AREA = "count_and_area"
    BASE_MASK_HISTO = "base_mask_histo"

    def dynargs(self):
        dynamic_args = {
            StatisticsMethod.GET_COUNT_AND_AREA.name: ['image_info', 'save'],
            StatisticsMethod.BASE_MASK_HISTO.name: ['base_res', 'image_info', 'save'],
        }
        return dynamic_args.get(self.name, [])

    def method(self, results=None, config: Config = None, args=None, dynargs=None, display=True, verbose=0):
        methods = {
            StatisticsMethod.GET_COUNT_AND_AREA.name: __get_count_and_area__,
            StatisticsMethod.BASE_MASK_HISTO.name: __mask_histogram_per_base_mask__
        }
        return methods[self.name] if results is None or args is None else methods[self.name](
            results, args, config, display, verbose, dynargs
        )
