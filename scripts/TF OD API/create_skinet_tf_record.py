import hashlib
import io
import logging
import os
from time import time

import numpy as np
import PIL
import contextlib2
import cv2
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_path', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_bool('no_shard', False, 'Number of TFRecord shards')
flags.DEFINE_string('log_path', None, 'Path used to save log file')

FLAGS = flags.FLAGS

dir2Skip = ['images', 'full_images']
UNSPECIFIED_POSE = 'unspecified'.encode('utf8')
FALSE_VALUE = int(False)
IMG_PER_SHARD = 400


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


def getMinMaxMaskCoordinates(maskImage):
    height, width = maskImage.shape
    yMin, xMin, yMax, xMax = extract_bboxes(maskImage)
    return xMin / float(width), xMax / float(width), yMin / float(height), yMax / float(height)


def getImageData(imageDirPath: str, label_map_dict, verbose=0):
    IMAGE_DATA = {
        'filename': None,
        'height': None,
        'width': None,
        'sha256': None,
        'encoded': None,
        'format': None,  # 'png'.encode('utf8'),
        'xmins': [],
        'ymins': [],
        'xmaxs': [],
        'ymaxs': [],
        'classes': [],
        'classes_text': [],
        'truncated': [],
        'poses': [],
        'difficult_obj': [],
        'masks': []
    }
    IMAGE_PATH = os.path.join(imageDirPath, 'images')
    IMAGE_PATH = os.path.join(IMAGE_PATH, os.listdir(IMAGE_PATH)[0])
    if '.png' in IMAGE_PATH:
        IMAGE_DATA['format'] = 'png'.encode('utf8')
    elif '.jpg' in IMAGE_PATH or '.jpeg' in IMAGE_PATH:
        IMAGE_DATA['format'] = 'jpeg'.encode('utf8')
    else:
        raise TypeError("Image must be either png or jpeg/jpg format.")
    image = cv2.imread(IMAGE_PATH)
    with tf.gfile.GFile(IMAGE_PATH, 'rb') as fid:
        encoded = fid.read()
    IMAGE_DATA['encoded'] = encoded
    IMAGE_DATA['sha256'] = hashlib.sha256(IMAGE_DATA['encoded']).hexdigest().encode('utf8')
    IMAGE_DATA['filename'] = os.path.basename(IMAGE_PATH).encode('utf8')
    IMAGE_DATA['height'], IMAGE_DATA['width'], _ = image.shape
    if verbose > 0:
        print('{} : {}x{}'.format(IMAGE_DATA['filename'], IMAGE_DATA['width'], IMAGE_DATA['height']))

    for class_name in os.listdir(imageDirPath):
        if class_name in label_map_dict:
            FOLDER_PATH = os.path.join(imageDirPath, class_name)
            for mask in os.listdir(FOLDER_PATH):
                MASK_PATH = os.path.join(FOLDER_PATH, mask)
                if verbose > 0:
                    print('\t' + MASK_PATH)
                mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
                pil_image = PIL.Image.fromarray(mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                IMAGE_DATA['masks'].append(output_io.getvalue())
                xmin, xmax, ymin, ymax = getMinMaxMaskCoordinates(mask)
                IMAGE_DATA['xmins'].append(xmin)
                IMAGE_DATA['xmaxs'].append(xmax)
                IMAGE_DATA['ymins'].append(ymin)
                IMAGE_DATA['ymaxs'].append(ymax)
                IMAGE_DATA['poses'].append(UNSPECIFIED_POSE)
                IMAGE_DATA['difficult_obj'].append(FALSE_VALUE)
                IMAGE_DATA['truncated'].append(FALSE_VALUE)
                IMAGE_DATA['classes_text'].append(class_name.encode('utf8'))
                IMAGE_DATA['classes'].append(label_map_dict[class_name])
    return IMAGE_DATA


def data2TFExample(imageData: dict):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(imageData['height']),
        'image/width': dataset_util.int64_feature(imageData['width']),
        'image/filename': dataset_util.bytes_feature(imageData['filename']),
        'image/source_id': dataset_util.bytes_feature(imageData['filename']),
        'image/key/sha256': dataset_util.bytes_feature(imageData['sha256']),
        'image/encoded': dataset_util.bytes_feature(imageData['encoded']),
        'image/format': dataset_util.bytes_feature(imageData['format']),
        'image/object/bbox/xmin': dataset_util.float_list_feature(imageData['xmins']),
        'image/object/bbox/xmax': dataset_util.float_list_feature(imageData['xmaxs']),
        'image/object/bbox/ymin': dataset_util.float_list_feature(imageData['ymins']),
        'image/object/bbox/ymax': dataset_util.float_list_feature(imageData['ymaxs']),
        'image/object/class/text': dataset_util.bytes_list_feature(imageData['classes_text']),
        'image/object/class/label': dataset_util.int64_list_feature(imageData['classes']),
        'image/object/difficult': dataset_util.int64_list_feature(imageData['difficult_obj']),
        'image/object/truncated': dataset_util.int64_list_feature(imageData['truncated']),
        'image/object/view': dataset_util.bytes_list_feature(imageData['poses']),
        'image/object/mask': dataset_util.bytes_list_feature(imageData['masks'])
    }))
    return example


def getNumShard(num_shards, index):
    return min(index // IMG_PER_SHARD, num_shards - 1)


'''
Lire XML
↳ Infos images
    - height
    - width
    - filename
    - source_id = filename
    - sha256
    - encoded
    - format
↳ Infos masques
    - Liste x, y mim max
    - Liste class label & text
    - Liste masques
        - numerical/image
'''


def main(_):
    start = time()
    LOG_FILE = FLAGS.log_path
    if LOG_FILE is not None:
        with open(LOG_FILE, 'w') as log:
            log.write("IMG_PATH\n")
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    DATASET_PATH = os.path.normpath(FLAGS.data_dir)
    imageDirList = os.listdir(DATASET_PATH)
    nbImage = len(imageDirList)
    record_dir = os.path.dirname(FLAGS.output_path)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir, exist_ok=True)
    num_shards = max(1, nbImage // IMG_PER_SHARD + (0 if nbImage % IMG_PER_SHARD < IMG_PER_SHARD * 0.2 else 1))
    if FLAGS.no_shard:
        num_shards = 1
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack,
                                                                                 FLAGS.output_path, num_shards)
        for idx, imageDir in enumerate(imageDirList):
            if LOG_FILE is not None:
                with open(LOG_FILE, 'a') as log:
                    log.write(imageDir + "\n")
            if idx % 50 == 0:
                logging.info('On image %d of %d', idx, len(imageDirList))

            IMAGE_DIR_PATH = os.path.join(DATASET_PATH, imageDir)
            data = getImageData(IMAGE_DIR_PATH, label_map_dict)
            tf_example = data2TFExample(data)
            output_tfrecords[idx % num_shards].write(tf_example.SerializeToString())
    total_time = time() - start
    m = int(total_time) // 60
    s = int(total_time) % 60
    print(f"{m:02d}:{s:02d}")


if __name__ == '__main__':
    tf.app.run()
