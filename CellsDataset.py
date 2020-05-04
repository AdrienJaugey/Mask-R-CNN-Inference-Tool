class CellsDataset(utils.Dataset):
    CELLS_CLASS_NAMES = ["tubule_sain", "tubule_atrophique", "nsg_complet", "nsg_partiel", "pac", "vaisseau", 
                         "artefact"]
    mode = ""
    def load_cells(self, mode):
        # Add classes
        self.add_class("cells", 1, "tubule_sain")
        self.add_class("cells", 2, "tubule_atrophique")
        self.add_class("cells", 3, "nsg_complet")
        self.add_class("cells", 4, "nsg_partiel")
        self.add_class("cells", 5, "pac")
        self.add_class("cells", 6, "vaisseau")
        self.add_class("cells", 7, "artefact")
        
        self.mode = mode
        if self.mode == "train":
            for n, id_ in enumerate(train_ids):
                path = TRAIN_PATH + id_
                img_path = path + '/images/'
                self.add_image("cells", image_id=id_, path=img_path)

        if self.mode == "val":
            for n, id_ in enumerate(map_ids):
                path = MAP_PATH + id_
                img_path = path + '/images/'
                self.add_image("cells", image_id=id_, path=img_path)

    def load_image(self, image_id):

        info = self.image_info[image_id]
        info = info.get("id")
        
        if self.mode == "train":
            path = TRAIN_PATH + info
        else:
            path = MAP_PATH + info
        img = imread(path + '/images/' + info + '.' + IMAGE_FORMAT)[:, :, :3]
        img = resize(img, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)

        return img

    def image_reference(self, image_id):
        """Return the cells data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cells":
            return info["cells"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for cells of the given image ID.
        """
        info = self.image_info[image_id]
        info = info.get("id")
        
        if self.mode == "train":
            path = TRAIN_PATH + info
        else:
            path = MAP_PATH + info
            
        # Counting masks for current image
        number_of_masks = 0
        for masks_dir in os.listdir(path):
            # For each directory excepting /images
            if masks_dir in ['images', 'full_images', 'cortex']:
                continue
            temp_DIR = path + '/' + masks_dir
            # Adding length of directory https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
            number_of_masks += len(
                [name for name in os.listdir(temp_DIR) if os.path.isfile(os.path.join(temp_DIR, name))])

        mask = np.zeros([config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], number_of_masks], dtype=np.uint8)
        iterator = 0
        class_ids = np.zeros((number_of_masks,), dtype=int)
        for masks_dir in os.listdir(path):
            if masks_dir in ['images', 'full_images', 'cortex']:
                continue
            temp_class_id = self.CELLS_CLASS_NAMES.index(masks_dir) + 1
            for mask_file in next(os.walk(path + '/' + masks_dir + '/'))[2]:
                mask_ = imread(path + '/' + masks_dir + '/' + mask_file)
                mask_ = resize(mask_, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant',
                               preserve_range=True)
                mask[:, :, iterator] = mask_
                class_ids[iterator] = temp_class_id
                iterator += 1
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(number_of_masks - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        return mask, class_ids.astype(np.int32)