"""
Skinet (Segmentation of the Kidney through a Neural nETwork) Project

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Written by Adrien JAUGEY
"""
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("label_map_proto_file", help="path to the label map file", type=str)
    parser.add_argument("tfrecord_path", type=str, help="path to the TFRecord files. You can use *.*-?????-of-00000 "
                                                        "format to loop through each shard")
    parser.add_argument("sleep_time", help="time before reading the next TFExample in ms", type=int, default=250,
                        nargs='?')
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    import cv2
    from object_detection.protos import input_reader_pb2
    from object_detection.data_decoders import tf_example_decoder as dec
    from object_detection.utils import visualization_utils as viz
    from object_detection.utils import label_map_util

    label_map_proto_file = os.path.normpath(args.label_map_proto_file)
    tfrecord_path = os.path.normpath(args.tfrecord_path)

    label_map_dict = label_map_util.get_label_map_dict(label_map_proto_file)
    category_index = {}
    for class_ in label_map_dict:
        category_index[label_map_dict[class_]] = {"id": label_map_dict[class_], "name": class_}
    # If TFRecord is sharded : get the number of shard and prepare path formatting
    nb_shard = 1
    if '-?????-of-' in tfrecord_path:
        nb_shard = int(tfrecord_path.split('-')[-1])
        tfrecord_path = tfrecord_path.replace('-?????-of-', '-{id_shard:05d}-of-')

    decoder = dec.TfExampleDecoder(
        True, input_reader_pb2.PNG_MASKS, use_display_name=True,
        label_map_proto_file=label_map_proto_file
    )

    for id_shard in range(nb_shard):
        for idx, record in enumerate(tf.data.TFRecordDataset(tfrecord_path.format(id_shard=id_shard))):
            decoded = decoder.decode(record)
            image = cv2.cvtColor(decoded['image'].numpy(), cv2.COLOR_RGB2BGR)
            image_name = decoded['filename'].numpy().decode('utf-8')
            boxes = decoded['groundtruth_boxes'].numpy()
            classes = decoded['groundtruth_classes'].numpy()
            masks = None
            if 'groundtruth_instance_masks' in decoded and decoded['groundtruth_instance_masks'].shape[0] > 0:
                masks = decoded['groundtruth_instance_masks'].numpy().astype(np.uint8)

            # Not really scores but do not matters
            scores = np.ones_like(classes)
            expected = viz.visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,
                                                                     instance_masks=masks, skip_scores=True,
                                                                     use_normalized_coordinates=True)
            height, width = image.shape[:2]
            ratio = 512 / max(height, width)
            expected = cv2.resize(expected, None, fx=ratio, fy=ratio)
            expected[0:18, 0:round(9.5 * len(image_name)), :] = (0, 0, 0)
            expected = cv2.putText(expected, image_name, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Image (press q to quit or p to toggle pause)', expected)
            key = None
            key = cv2.waitKey(args.sleep_time) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord('p'):
                key = None
                while key not in [ord('p'), ord('q')]:
                    key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(0)
    cv2.destroyAllWindows()
    exit(0)
