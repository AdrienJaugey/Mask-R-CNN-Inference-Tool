import os
import numpy as np
from skimage.io import imread

from datasetTools.datasetWrapper import getBboxFromName
from mrcnn import utils


class SkinetCustomDataset(utils.Dataset):

    def __init__(self, custom_class_names, mode, cortex_size, config, image_info,
                 enable_occlusion=False):
        super().__init__()
        self.__CUSTOM_CLASS_NAMES = custom_class_names.copy()
        self.__MODE = mode
        self.__CORTEX_SIZE = cortex_size
        self.__CONFIG = config
        self.__IMAGE_INFO = image_info
        self.__ENABLE_OCCLUSION = enable_occlusion

    def get_class_names(self):
        return self.__CUSTOM_CLASS_NAMES.copy()

    def get_visualize_names(self):
        visualize_names = ['background']
        visualize_names.extend(self.__CUSTOM_CLASS_NAMES)
        return visualize_names

    def load_images(self):
        # Add classes
        for class_id, class_name in enumerate(self.__CUSTOM_CLASS_NAMES):
            self.add_class("skinet", class_id + 1, class_name)
        image_name = self.__IMAGE_INFO["NAME"]
        img_path = os.path.join('data', image_name, "images",
                                f"{image_name}.{self.__IMAGE_INFO['IMAGE_FORMAT']}")
        self.add_image("skinet", image_id=self.__IMAGE_INFO["NAME"], path=img_path)

    def image_reference(self, image_id_):
        """Return the skinet data of the image."""
        info = self.image_info[image_id_]
        if info["source"] == "skinet":
            return info["skinet"]
        else:
            super(self.__class__).image_reference(self, image_id_)

    def load_mask(self, image_id_):
        """Generate instance masks for cells of the given image ID.
        """
        info = self.image_info[image_id_]
        info = info.get("id")

        path = os.path.join('data', info)

        # Counting masks for current image
        number_of_masks = 0
        for masks_dir in os.listdir(path):
            # For each directory excepting /images
            if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                continue
            temp_DIR = os.path.join(path, masks_dir)
            # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
            number_of_masks += len([name_ for name_ in os.listdir(temp_DIR)
                                    if os.path.isfile(os.path.join(temp_DIR, name_))])
        if self.__MODE == "cortex":
            masks_shape = (self.__CORTEX_SIZE[0], self.__CORTEX_SIZE[1], number_of_masks)
        elif self.__CONFIG.USE_MINI_MASK:
            masks_shape = (self.__CONFIG.MINI_MASK_SHAPE[0], self.__CONFIG.MINI_MASK_SHAPE[1],
                           number_of_masks)
        else:
            masks_shape = (self.__IMAGE_INFO["HEIGHT"], self.__IMAGE_INFO["WIDTH"], number_of_masks)
        masks = np.zeros(masks_shape, dtype=np.uint8)
        bboxes = np.zeros((number_of_masks, 4), dtype=np.int32)
        iterator = 0
        class_ids = np.zeros((number_of_masks,), dtype=int)
        for masks_dir in os.listdir(path):
            if masks_dir not in self.__CUSTOM_CLASS_NAMES:
                continue
            temp_class_id = self.__CUSTOM_CLASS_NAMES.index(masks_dir) + 1
            masks_dir_path = os.path.join(path, masks_dir)
            for mask_file in os.listdir(masks_dir_path):
                mask = imread(os.path.join(masks_dir_path, mask_file))
                masks[:, :, iterator] = mask
                if self.__CONFIG.USE_MINI_MASK:
                    bboxes[iterator] = getBboxFromName(mask_file)
                else:
                    bboxes[iterator] = utils.extract_bboxes(mask)
                class_ids[iterator] = temp_class_id
                iterator += 1
        # Handle occlusions /!\ In our case there is no possible occlusion (part of object that
        # is hidden), all objects are complete (some are parts of other)
        if self.__ENABLE_OCCLUSION:
            occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
            for i in range(number_of_masks - 2, -1, -1):
                masks[:, :, i] = masks[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
        return masks, class_ids.astype(np.int32), bboxes
