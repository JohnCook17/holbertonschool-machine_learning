#!/usr/bin/env python3
"""You only look once"""
import numpy as np
import tensorflow as tf


class Yolo:
    """The Yolo class"""
    def __init__(self,
                 model_path,
                 classes_path,
                 class_t,
                 nms_t,
                 anchors):
        """initialization of the yolo class"""
        self.__model_path = model_path
        self.__classes_path = classes_path
        self.__class_t = class_t
        self.__nms_t = nms_t
        self.__anchors = anchors

    @property
    def model(self):
        """returns the model of yolo"""
        return tf.keras.models.load_model(self.__model_path)

    @property
    def class_names(self):
        """returns a list of class names for yolo"""
        with open(self.__classes_path) as my_file:
            my_classes = my_file.read().splitlines()
        return my_classes

    @property
    def class_t(self):
        """returns the threshold of a class"""
        return self.__class_t

    @property
    def nms_t(self):
        """returns the intersection over union threshold"""
        return self.__nms_t

    @property
    def anchors(self):
        """contains all the anchors"""
        return self.__anchors

    def process_outputs(self, outputs, image_size):
        """Processed outputs"""
        boxes = []
        box_confidence = []
        box_class_probs = []
        for output in outputs:
            image_height = image_size[0]
            image_width = image_size[1]
            anchor_boxes_shape = output.shape[2]
            classes_shape = output.shape[3]
            output = np.resize(output,
                               (image_height,
                                image_width,
                                anchor_boxes_shape,
                                classes_shape))
            boxes.append(output[:, :, :, 0: 4])
            box_confidence.append(output[:, :, :, 4: 5])
            box_class_probs.append(output[:, :, :, 5:])

        return boxes, box_confidence, box_class_probs
