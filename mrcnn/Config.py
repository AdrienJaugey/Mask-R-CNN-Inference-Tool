"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import json
from abc import abstractmethod
from enum import Enum


class Config:
    """
    Wraps the config dictionary with easy access methods
    """

    def __init__(self, config, mode: str = None, forceFullSizeMasks: bool = False):
        """
        Instanciates Config object using given path or dictionary
        :param config: path to the config json file or complet raw config dictionary
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :param forceFullSizeMasks: if True, will force full-sized masks
        """
        self.__BASE_CONFIG__ = config
        if type(config) is str:
            with open(config, 'r') as configFile:
                self.__CONFIG_DICT__ = json.load(configFile)
        else:
            self.__CONFIG_DICT__ = config.copy()
        self.__init_modes__()
        self.__init_classes_info__()
        self.__init_classes_hierarchy__()
        self.__CURRENT_MODE__ = mode
        self.__FORCE_FULL_MASKS = forceFullSizeMasks

    def copy(self):
        return Config(self.__BASE_CONFIG__, mode=self.__CURRENT_MODE__, forceFullSizeMasks=self.__FORCE_FULL_MASKS)

    def __init_modes__(self):
        """
        Transforms modes array into a dict with modes indexed by their "name" value
        :return: None
        """
        self.__CONFIG_DICT__['modes'] = {m['name']: m for m in self.__CONFIG_DICT__['modes']}

    def set_current_mode(self, mode: str, forceFullSizeMasks: bool = None):
        """
        Sets the current mode to enable accessing this mode info without passing it to other methods
        :param mode: the mode to set
        :param forceFullSizeMasks: if True, will force full-sized masks
        :return: None
        """
        self.__CURRENT_MODE__ = mode
        if forceFullSizeMasks is not None:
            self.__FORCE_FULL_MASKS = forceFullSizeMasks

    def set_force_full_masks(self, value: bool):
        self.__FORCE_FULL_MASKS = value

    def get_current_mode(self):
        """
        Returns the current mode set
        :return: current mode
        """
        return self.__CURRENT_MODE__

    def get_previous_mode(self, mode: str = None):
        """
        Returns the mode that is executed before the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: If existing, the mode as str
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('previous', None)

    def has_previous_mode(self, mode: str = None):
        """
        Returns whether the given/current mode has a previous mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        return self.get_previous_mode(mode) is not None

    def get_next_mode(self, mode: str = None):
        """
        Returns the mode that is executed after the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: If existing, the mode as str
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('next', None)

    def has_next_mode(self, mode: str = None):
        """
        Returns whether the given/current mode has a previous mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        return self.get_next_mode(mode) is not None

    def get_chain_order(self):
        """
        Returns an ordered list of mode names from first mode to 
        :return: list of mode name
        """
        mode_ = self.get_first_mode()
        order = []
        while mode_ is not None:
            order.append(mode_)
            mode_ = self.get_next_mode(mode_)
        return order

    def get_mode_config(self, mode: str = None):
        """
        Returns the raw config dictionary of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: raw config as a dict
        """
        special_modes = {
            "first": self.get_first_mode,
            "previous": self.get_previous_mode,
            "next": self.get_next_mode
        }
        mode_ = self.__CURRENT_MODE__ if mode is None or mode == "current" else mode
        if mode_ in special_modes:
            mode_ = special_modes[mode_]()
        return self.__CONFIG_DICT__['modes'].get(mode_, None)

    def get_mode_list(self):
        """
        Returns all the available mode names
        :return: mode names as a list
        """
        return list(self.__CONFIG_DICT__['modes'].keys())

    def get_first_mode(self):
        """
        Returns the name of the first mode of 'chain mode' if specified, else returns first mode in the list
        :return: name of the first mode as str
        """
        return self.__CONFIG_DICT__.get('first_mode', self.get_mode_list()[0])

    def get_mode_name(self, mode=None):
        """
        Returns the name of the given/current mode if it exists
        :return: name of the given/current mode or None
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config['name']

    def __init_classes_info__(self):
        """
        Adds id to classes and generates name to id dictionary
        :return: None
        """
        for mode in self.get_mode_list():
            mode_config = self.get_mode_config(mode)
            mode_config['class_to_id'] = {}
            for idx, classInfo in enumerate(mode_config['classes']):
                _idx = idx + 1
                classInfo['id'] = _idx
                mode_config['class_to_id'][classInfo['name']] = _idx

    def get_classes_info(self, mode: str = None):
        """
        Returns dictionary with classes indexed by id of the give/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: classes info as dict
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is None:
            return None
        return mode_config.get('classes', None)

    def get_class_id(self, class_name: str, mode: str = None):
        """
        Returns id of the class of the given name for the given/current mode
        :param class_name: name of the class
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: id of the given class
        """
        if type(class_name) is int:
            return class_name
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config['class_to_id'].get(class_name, -1)
        return -1

    def get_class_name(self, class_id: int, mode: str = None, display=False):
        """
        Returns name of the class of the given id for the given/current mode
        :param class_id: id of the class
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :param display: if True, will return the display name instead of the base name, if it exists
        :return: name of the given class
        """
        if type(class_id) is str:
            return class_id
        classes_info = self.get_classes_info(mode)
        if classes_info is not None and class_id - 1 < len(classes_info):
            selected_class = classes_info[class_id - 1]
            if display and 'display_name' in selected_class:
                return selected_class['display_name']
            return selected_class['name']

    def __init_classes_hierarchy__(self):
        """
        Initializes classes hierarchy of each mode
        :return: None
        """
        for mode in self.get_mode_list():
            isTherePriority = max([c.get('priority_level', 0) for c in self.get_classes_info(mode)]) > 0
            isThereContains = any(['contains' in c for c in self.get_classes_info(mode)])
            hierarchy = None
            if isThereContains and not isTherePriority:
                hierarchy = [
                    ({c['id']: [self.get_class_id(c2, mode) for c2 in c['contains']]} if 'contains' in c else c['id'])
                    for c in self.get_classes_info(mode)
                ]
                if not hierarchy:
                    hierarchy = None
            elif isTherePriority and not isThereContains:
                lvl = {}
                for c in self.get_classes_info(mode):
                    priority = c.get('priority_level', 0)
                    if priority not in lvl:
                        lvl[priority] = []
                    lvl[priority].append(c['id'])
                keys = list(lvl.keys())
                keys.sort(reverse=True)
                first = True
                for key in keys:
                    if first:
                        first = False
                        hierarchy = lvl[key]
                    else:
                        if len(lvl[key]) == 1:
                            hierarchy = {lvl[key][0]: hierarchy}
                        else:
                            lvl[key][-1] = {lvl[key][-1]: hierarchy}
                            hierarchy = lvl[key]
            mode_config = self.get_mode_config(mode)
            mode_config['classes_hierarchy'] = hierarchy

    def get_classes_hierarchy(self, mode: str = None):
        """
        Returns class_hierarchy of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: classes hierarchy of the given/current mode
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('classes_hierarchy', None)

    def has_class_name(self, class_name: str, mode: str = None):
        """
        Returns True if given/current mode has a class named class_name
        :param class_name: name of the class
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: True if class is from this mode
        """
        return self.get_class_id(class_name, mode) != -1

    def get_class_mode(self, class_name: str, only_in_previous: str = None):
        """
        Returns the modes containing this class
        :param class_name: name of the class
        :param only_in_previous: If given, specifies the mode for which you want the info. Special modes: first,
                                 previous, current, next
        :return: list of modes containing the class
        """
        if only_in_previous is not None:
            res = []
            while only_in_previous is not None:
                if self.has_class_name(class_name, only_in_previous):
                    res.append(self.get_mode_name(only_in_previous))
                only_in_previous = self.get_previous_mode(only_in_previous)
            return res
        return [mode for mode in self.get_mode_list() if self.has_class_name(class_name, mode)]

    def get_param(self, mode: str = None):
        """
        Returns the parameters of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: dict of parameters
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('parameters')

    def is_using_mini_mask(self, mode: str = None):
        """
        Returns True if given/current mode uses mini-masks
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        if self.__FORCE_FULL_MASKS:
            return False
        mode_param = self.get_param(mode)
        if mode_param is not None:
            return mode_param.get('mini_mask', None) is not None

    def get_mini_mask_size(self, mode: str = None):
        """
        Returns mini-mask size of the given/current mode if specified, else None
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: mini-masks size as int if used
        """
        if self.is_using_mini_mask(mode):
            return self.get_param(mode)['mini_mask']

    def get_mini_mask_shape(self, mode: str = None):
        """
        Returns mini-mask size of the given/current mode if specified, else None
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: mini-masks shape as int if used
        """
        if self.is_using_mini_mask(mode):
            return (self.get_param(mode)['mini_mask'],) * 2

    def has_to_export(self, mode: str = None):
        """
        Returns True if given/current mode exports its results as annotations
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('export', None) is not None

    def get_export_param(self, mode: str = None):
        """
        Returns export parameter of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: export parameters as str or str list
        """
        if self.has_to_export(mode):
            return self.get_mode_config(mode)['export']

    def has_to_export_cleaned_img(self, mode: str = None):
        """
        Returns True if given/current mode has to exports a cleaned image
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('export_cleaned_image', None) is not None

    def get_export_param_cleaned_img(self, mode: str = None):
        """
        Returns cleaned image export parameters of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: export parameters as a dict
        """
        if self.has_to_export(mode):
            return self.get_mode_config(mode)['export_cleaned_image']

    def has_to_return(self, mode: str = None):
        """
        Returns True if given/current mode returns at least a part of its results
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: bool
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get('return', None) is not None

    def get_return_param(self, mode: str = None):
        """
        Returns return parameter of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: export parameters as str or str list
        """
        if self.has_to_return(mode):
            return self.get_mode_config(mode)['return']

    def __get_dynamic_methods__(self, methodType: str, mode: str = None):
        """
        Returns the dynamic methods list of the given method type for the given/current mode
        :param methodType: The type of method to get in: post_processing, statistics
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: dynamic methods array
        """
        mode_config = self.get_mode_config(mode)
        if mode_config is not None:
            return mode_config.get(methodType, [])

    def get_post_processing_method(self, mode: str = None):
        """
        Returns the dynamic methods list of post-processing methods of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: dynamic post-processing methods as list
        """
        return self.__get_dynamic_methods__('post_processing', mode)

    def get_statistics_method(self, mode: str = None):
        """
        Returns the dynamic methods list of statistics methods of the given/current mode
        :param mode: If given, specifies the mode for which you want the info. Special modes: first, previous, next
        :return: dynamic statistics methods as list
        """
        return self.__get_dynamic_methods__('statistics', mode)


class DynamicMethod(Enum):

    @abstractmethod
    def dynargs(self):
        pass

    @abstractmethod
    def method(self, results=None, config: Config = None, args=None,
               dynargs=None, display=True, verbose=0):
        pass
