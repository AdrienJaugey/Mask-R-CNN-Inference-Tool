# Writing a configuration file

​	The configuration file is used to define the inference tool's behavior, it can contain multiple inference modes and those can be chained by defining the data that a mode has to provide to the following one. This file is required as the Inference Tool will not know what to do without it. The configuration file is a simple JSON file 

​	You can find a configuration file example in this documentation [here](configuration_example.json). This example will be used and commented in this page.

## Index

1. [Mode and mode's order definition](#1-mode-and-modes-order-definition)
2. [Classes definition](#2-classes-definition)
3. [Pre-inference and Inference parameters](#3-pre-inference-and-inference-parameters)
4. [Post-processing methods](#4-post-processing-methods)
5. [Statistics methods](#5-statistics-methods)
6. [Additional results files](#6-additional-results-files)

## 1. Mode and mode's order definition

```json
{
    "first_mode": "mode1",
    "modes": [
        {
            "name": "mode1",
            "previous": null,
            "next": "mode2",
            ...
        },
        {
            "name": "mode2",
            "previous": "mode1",
            "next": null,
            ...
        }
}
```

​	There are two main elements at the root of our configuration file. 

​	The first one is `first_mode` that stores the name of the first mode to execute when the tool is in chain mode (modes will be executed one after the other, results will be transmitted so that they can be used to select areas for the inference or to get specific statistics).

> In the example, the first mode to execute is "mode1".

​	The second element is the array `modes` that will contains dictionaries describing each modes (one per mode). Each mode has a `name` and knows the name of the `previous` and `next` modes (`null` if there is not a previous/next mode). 

> In the example, there are two modes : "mode1" and "mode2", "mode1" (resp. "mode2") is the mode executed after (resp. before) "mode1" (resp. "mode2").


## 2. Classes definition

```json
{
    "name": "mode1",
    ...
    "classes": [
        {
            "name": "class1",
            "display_name": "Class 1",
            "asap_color": "#ffaa00",
            "priority_level": 1,
            "color": [255, 0, 0]
        },
        {
            "name": "class2",
            "display_name": "Class 2",
            "asap_color": "#ff0000",
            "color": [0, 255, 0]
        },
        {
            "name": "class3",
            "display_name": "Class 3",
            "asap_color": "#ffffff",
            "color": [0, 0, 255]
        }
    ],
    ...
}
```

​	Obviously, the Inference Tool has to know the different classes of object of each mode. To do that, a `classes` array inside each mode dictionary (cf. [§1](#1-mode-and-modes-order-definition)), this array contains dictionaries that describes each class of the corresponding mode.

​	A class is described by :

- a `name` which could be a short version of the name without space; 

- a `display_name` which will be used on masked images and other results file(s). It can be a complete/scientific name of the object class;

- a `color` which will be used when mask of the corresponding class is applied on the image. The color is given as an array of three integers values between 0 and 255, corresponding to red, green and blue channels;

- [OPTIONAL] an `asap_color` which is an RGB color given as a hexadecimal string and that is used when exporting as an ASAP annotations file (classes in ASAP format have a specific color);

- [OPTIONAL] Hierarchy/Ordering of the classes :

    - a `priority_level` : an integer representing the importance of the corresponding class, the higher it is, the more prioritized the class is. If omitted, classes have a priority level of 0.

        OR (only one type of hierarchy/ordering for a mode) 

    - a `contains` array : which contains the names of the classes (subclasses actually) that belong to the class (for example, a "wheel" mask should be included in a "car" mask). Can be omitted for classes that do not contains other classes. You can find an example of this hierarchy/ordering parameter in the [example configuration file](configuration_example.json#L69).

    These parameters will be used in some methods where a pixel cannot be associated to multiple classes to ensure that the most prioritizes classes are kept for conflicting cases.

> In the example, the "class1" class will be represented in red color with "Class 1" as display name, and has a higher priority level than "class2" and "class3".

## 3. Pre-inference and Inference parameters

```json
{
    "name": "mode2",
    ...
    "parameters": {
        "weight_file": "path/to/weights_folder_of_mode2",
        "base_class": "class1",
        "exclude_class": null,
        "fuse_base_class": true,
        "crop_to_remaining": true,
        "allow_empty_annotations": false,
        "roi_mode": "divided",
        "resize": null,
        "roi_size": 1024,
        "min_overlap_part": 0.33,
        "mini_mask": 128,
        "min_confidence": 0.5,
        "allow_sparse": false
    },
    ...
}
```

​	Each inference mode has a `parameters` dictionary used to define variables needed by the tool for loading the AI, preparing the image before inference (keeping or excluding specific parts for example, retrieving information about the image, etc... ), selecting parts of the image that will be processed by the AI, or formatting the results. 

Those parameters can be :

- a `weight_file` string, representing the path to the weights file/folder of the mode. It can be relative to the repository folder;
- a `base_class` string or strings array, represent the class(es) name(s) used as a base for the image (kept part of an image), those are classes from annotations file or previous modes if using chain mode (and those modes transmit the corresponding classes). If there is no base class, set the value to `null`;

> You can even use the same base class used multiples modes before if all modes between are also using it and then transferring it, simply put its class name.

- an `exclude_class` string or strings array, same thing as `base_class` but it represents excluded part(s) of the image;

> If you set `exclude_class` to `all`, you can use same base of previous modes, else the base will be assumed as the entire image.

- a `fuse_base_class` boolean that defines whether the base class(es) will be fused or not;

- a `crop_to_remaining` boolean that defines whether the image will be cropped to keep only the remaining part(s) of the image or not;

- an `allow_empty_annotations` boolean that defines whether cases where the tool has to predict nothing are allowed or not;

- an `roi_mode` string that will define how the tool will select part of the image to pass to the AI. This parameter can be set to :

    - `divided` : the image will be divided in square parts of `roi_size` side size with at least `min_overlap_part` overlapping between a division and the following one.

        OR

    - `centered` : the tool will select rectangle parts of a length/width of at least `roi_size` and centered on each masks of base class(es).

-  a `resize` integers array which defines the size (height, width) that will be used to resize the input image. If you do not want the input image to be resized, set this parameter to `null`;

- an `roi_size` integer that defines the side (resp. least side size) of the image part passed to the AI when `roi_mode` is set to `divided` (resp. `centered`);

> For example, if you want the input image to be resized to a 1920 x 1080 image, set the `resize` parameter to `[1080, 1920]` .

- a `min_overlap_part` float between 0.0 and 1.0 that defines the minimum overlap between two divisions when `roi_mode` is set to `divided`;
- a `mini_mask` positive integer that defines the size of the mini-masks (used to process high resolution images that would take too much memory with full-sized masks). If you do not want mini-masks to be used, set this parameter to `null`.
- a `min_confidence` float between 0.0 and 1.0. When the AI makes predictions, every masks with a score greater or equal to this value will be kept;
- an `allow_sparse` boolean that defines whether only the biggest part of a mask is kept (`false`) or not (`true`). 

## 4. Post-processing methods

```json
{
    "name": "mode2",
    ...
    "post_processing": [
        {
            "method": "fusion",
            "bb_threshold": 0.1,
            "mask_threshold": 0.1
        },
        ...
        {
            "method": "filter",
            "bb_threshold": 0.3,
            "mask_threshold": 0.3,
            "priority_table": [
                [false, true,  false, true ],
                [false, false, false, false],
                [true,  true,  false, true ],
                [false, false, false, false]
            ]
        }
        ...
    ],
    ...
}
```

​	A `post_processing` array is found inside each mode configuration and defines which post-processing methods have to be applied to the results. Those methods are described as dictionaries with a `method` string corresponding to the method's name and additional parameters that will define the method behavior. Those additional parameters are described in the following table and can be added in the same way as in the example :

<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">Method</th>
    <th class="tg-uzvj">Description</th>
    <th class="tg-uzvj">Parameter</th>
    <th class="tg-uzvj">Description</th>
    <th class="tg-uzvj">Type [default value]</th>
    <th class="tg-uzvj">Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-uzvj" rowspan="2">fusion</td>
    <td class="tg-9wq8" rowspan="2">Fuses overlapping masks of the same class</td>
    <td class="tg-9wq8">bb_threshold</td>
    <td class="tg-9wq8">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.1]</span></td>
    <td class="tg-9wq8">0.33</td>
  </tr>
  <tr>
    <td class="tg-9wq8">mask_threshold</td>
    <td class="tg-9wq8">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.1]</span></td>
    <td class="tg-9wq8">0.5</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">class_fusion</td>
    <td class="tg-9wq8" rowspan="3">Fuses overlapping masks of different classes, keeping the highest score class</td>
    <td class="tg-9wq8">bb_threshold</td>
    <td class="tg-9wq8">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.1]</span></td>
    <td class="tg-9wq8">0.33</td>
  </tr>
  <tr>
    <td class="tg-9wq8">mask_threshold</td>
    <td class="tg-9wq8">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.1]</span></td>
    <td class="tg-9wq8">0.5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">classes_compatibility</td>
    <td class="tg-9wq8">List of groups of classes' IDs that can be merged together</td>
    <td class="tg-9wq8">Array of int arrays</td>
    <td class="tg-9wq8">[[1, 2]]</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">filter</td>
    <td class="tg-9wq8" rowspan="3">Filters masks based on class-to-class rules (i.e. if a mask of class A is overlapping mask of class B, mask of class A will be removed)</td>
    <td class="tg-9wq8">bb_threshold</td>
    <td class="tg-9wq8">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.5]</span></td>
    <td class="tg-9wq8">0.33</td>
  </tr>
  <tr>
    <td class="tg-9wq8">mask_threshold</td>
    <td class="tg-9wq8">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.2]</span></td>
    <td class="tg-9wq8">0.5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">priority_table</td>
    <td class="tg-9wq8">An element set to true in the n-th column and m-th row means that a mask of the class with ID n, if contained by a mask of the class with ID m, will be erased</td>
    <td class="tg-9wq8">N² matrix of boolean with N being the number of classes</td>
    <td class="tg-9wq8">[[false, true, false, true],<br> [false, false, false, false],<br> [true, true, false, true],<br> [false, false, false, false]]</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">orphan_filter</td>
    <td class="tg-9wq8" rowspan="3">Filter that ensures that some children-classes' masks are presents in a parent-class mask, and, if enabled, ensures that the inverse is also true</td>
    <td class="tg-c3ow">bb_threshold</td>
    <td class="tg-c3ow">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.5]</span></td>
    <td class="tg-c3ow">0.33</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mask_threshold</td>
    <td class="tg-c3ow">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.5]</span></td>
    <td class="tg-c3ow">0.5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">classes_hierarchy</td>
    <td class="tg-9wq8">Defines the parent and children classes and if parent class can exist without a child-class' mask ('keep_if_no_child' is true) or not (false)</td>
    <td class="tg-9wq8">{&lt;parent-class ID&gt;: {'contains': [children-classes' ID], 'keep_if_no_child': boolean}}</td>
    <td class="tg-9wq8">{"3": {"contains": [4, 5], <br>"keep_if_no_child": false},<br>"8": {"contains": [9, 10], <br>"keep_if_no_child": true}}</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">small_filter</td>
    <td class="tg-9wq8" rowspan="2">Filter that removes masks with less than a specific area (can be applied to only some classes)</td>
    <td class="tg-9wq8">min_size</td>
    <td class="tg-9wq8">The minimal pixel area for a mask to be kept</td>
    <td class="tg-9wq8">Positive integer <span style="font-weight:bold">[300]</span></td>
    <td class="tg-9wq8">50</td>
  </tr>
  <tr>
    <td class="tg-9wq8">classes</td>
    <td class="tg-9wq8">If given, the ID of the classes that are concerned by the filter</td>
    <td class="tg-9wq8">List of integer <span style="font-weight:bold">[None]</span></td>
    <td class="tg-9wq8">[1, 2, 4]</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">border_filter</td>
    <td class="tg-9wq8" rowspan="2">Filter that removes masks that are too much overlapping with the border/void part of a cleaned image</td>
    <td class="tg-9wq8">on_border_threshold</td>
    <td class="tg-9wq8">The least part of a mask to be on the border/void (#000000 color) part of the image for it to be deleted</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.25]</span></td>
    <td class="tg-9wq8">0.42</td>
  </tr>
  <tr>
    <td class="tg-c3ow">classes</td>
    <td class="tg-c3ow">If given, the ID of the classes that are concerned by the filter</td>
    <td class="tg-c3ow">List of integer <span style="font-weight:bold">[None]</span></td>
    <td class="tg-c3ow">[1, 2, 4]</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="4">exclude_masks_part</td>
    <td class="tg-9wq8" rowspan="4">Removing common part of masks that are overlapping with specific classes' ones</td>
    <td class="tg-c3ow">bb_threshold</td>
    <td class="tg-c3ow">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.5]</span></td>
    <td class="tg-c3ow">0.33</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mask_threshold</td>
    <td class="tg-c3ow">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.2]</span></td>
    <td class="tg-c3ow">0.5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">confidence_threshold</td>
    <td class="tg-9wq8">If given, the minimal score of an overlapping mask to extrude the other mask. If not given, the overlapping mask's score has to be greater or equal than the other mask's score</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 or None <span style="font-weight:bold">[0.7]</span></td>
    <td class="tg-9wq8">0.36</td>
  </tr>
  <tr>
    <td class="tg-c3ow">classes</td>
    <td class="tg-c3ow">Defines the classes that can be extruded and the class(es) than could extrude them</td>
    <td class="tg-c3ow">{&lt;ID or name of a class that could be extruded&gt;: [&lt;IDs or names of the classes that would extrude if overlapping&gt;]}</td>
    <td class="tg-c3ow">{'class1: ['class2', 'class3'], 4: [5, 6]}</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">keep_biggest_mask</td>
    <td class="tg-9wq8" rowspan="3">Filter that keep only the biggest mask between two overlapping ones</td>
    <td class="tg-c3ow">bb_threshold</td>
    <td class="tg-c3ow">Least part of bounding boxes overlapping to consider the two masks can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.5]</span></td>
    <td class="tg-c3ow">0.33</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mask_threshold</td>
    <td class="tg-c3ow">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-c3ow">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.2]</span></td>
    <td class="tg-c3ow">0.5</td>
  </tr>
  <tr>
    <td class="tg-9wq8">classes</td>
    <td class="tg-9wq8">Defines the classes to test at the same time with this filter</td>
    <td class="tg-9wq8">{&lt;ID or name of a class to test&gt;: [&lt;ID(s) or name(s) of class(es) to test at the same time&gt;]}</td>
    <td class="tg-c3ow">{'class1: ['class2', 'class3'], 4: [5, 6]}</td>
  </tr>
</tbody>
</table>




## 5. Statistics methods

```json
{
    "name": "mode2",
    ...
    "statistics": [
        {
            "method": "base_mask_histo",
            "classes": {"class1": ["all"]},
            "box_epsilon": 0,
            "test_masks": true,
            "count_zeros": true,
            "mask_threshold": 0.9,
            "display_per_base_mask": false,
            "display_global": true
        }
    ],
    ...
}
```

​	A `statistics` array is found inside each mode configuration and defines which statistics methods will be applied to the results, in the same way as post-processing methods. They are only 2 statistics methods for now :

<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">Method</th>
    <th class="tg-uzvj">Description</th>
    <th class="tg-uzvj">Parameter</th>
    <th class="tg-uzvj">Description</th>
    <th class="tg-uzvj">Type [default value]</th>
    <th class="tg-uzvj">Example</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-uzvj">count_and_area</td>
    <td class="tg-9wq8">Get count and total area of each selected classes</td>
    <td class="tg-9wq8">selected_classes</td>
    <td class="tg-9wq8">The selected class(es) to get statistics from</td>
    <td class="tg-9wq8">List of name or IDs of the classes, or "all" <span style="font-weight:bold">["all"]</span></td>
    <td class="tg-9wq8">["class1", "class3"]</td>
  </tr>
  <tr>
    <td class="tg-wa1i" rowspan="7">base_mask_histo</td>
    <td class="tg-nrix" rowspan="7">Computes a histogram of the selected-class(es)' masks inside a previous or current selected-class(es)</td>
    <td class="tg-nrix">classes</td>
    <td class="tg-nrix">Defines which classes (from current mode) to search inside which classes (from current or previous modes).</td>
    <td class="tg-nrix">{&lt;name of base class&gt;: [&lt;name of classes to count inside base&gt;]}</td>
    <td class="tg-nrix">{'class1': ['otherClass1', 'otherClass2']}</td>
  </tr>
  <tr>
    <td class="tg-nrix">box_epsilon</td>
    <td class="tg-nrix">Additional margin to allow masks with a bounding box with a maximum offset of box_epsilon pixels</td>
    <td class="tg-nrix">Positive integer <span style="font-weight:bold">[0]</span></td>
    <td class="tg-nrix">20</td>
  </tr>
  <tr>
    <td class="tg-nrix">test_masks</td>
    <td class="tg-nrix">Whether to test if mask is included into the base one or not</td>
    <td class="tg-nrix">Boolean <span style="font-weight:bold">[True]</span></td>
    <td class="tg-nrix">true</td>
  </tr>
  <tr>
    <td class="tg-nrix">count_zeros</td>
    <td class="tg-nrix">Whether to count base masks that does not have a specific class inside them</td>
    <td class="tg-baqh">Boolean <span style="font-weight:bold">[True]</span></td>
    <td class="tg-nrix">false</td>
  </tr>
  <tr>
    <td class="tg-nrix">mask_threshold</td>
    <td class="tg-nrix">Least part of masks overlapping to consider they can affect each other</td>
    <td class="tg-nrix">Float between 0.0 and 1.0 <span style="font-weight:bold">[0.9]</span></td>
    <td class="tg-nrix">0.6</td>
  </tr>
  <tr>
    <td class="tg-nrix">display_per_base_mask</td>
    <td class="tg-nrix">Whether to display the histogram of each base class or not (not recommended if there are a lot of base masks)</td>
    <td class="tg-baqh">Boolean <span style="font-weight:bold">[False]</span></td>
    <td class="tg-nrix">false</td>
  </tr>
  <tr>
    <td class="tg-nrix">display_global</td>
    <td class="tg-nrix">Whether to display the global histogram (sort of summary of all the histograms)</td>
    <td class="tg-baqh">Boolean <span style="font-weight:bold">[True]</span></td>
    <td class="tg-nrix">true</td>
  </tr>
</tbody>
</table>


## 6. Additional results files

```json
{
    "name": "mode1",
    ...
    "export": "all",
    "export_cleaned_image": [
        {
            "name": "class1",
            "crop_to_remaining": true,
            "base_class": "class1",
            "exclude_class": null
        }
    ],
    "return": ["class1"]
}
```

​	Each mode can generate additional results files such as annotations files (from the supported formats), and cleaned images (an image that has only some parts/masks kept based on predictions). A mode can also pass some of its results to the following mode for it to use them as base/excluded parts or for histogram calculation. Those parameters are the following :

- an `export` string or array of strings that tells the tool which annotation format to export to, value can be `"all"`, `"ASAP"` (case insensitive), `"LabelMe"` (case insensitive), or whatever additional format that was added;
- an `export_cleaned_image` array containing dictionaries that describe each cleaned image that can export the tool. A dictionary has the following parameters:
    - a `name` string : if given, will be used in the name of the resulting image file, else a number will be affected as the name to differentiate multiple cleaned images;
    - a `crop_to_remaining` boolean that defines whether the tool will crop the image to the minimum area containing all the kept parts;
    - a `base_class` string or array of strings that defines which class(es) from the current mode (or previous mode if you want to reuse same base class as the current mode, for that, use same name as base class or `"base"` value) are used as a base (kept part of the image);
    - a `exclude_class` string or array of strings : same as `base_class` but only from current mode, represents excluded parts of the cleaned image. Value can be `"all"`.
- a `return` string or array of strings, representing the class(es) to pass to the following mode. Values can be `"all"`, `"base"` and the name of current classes.
