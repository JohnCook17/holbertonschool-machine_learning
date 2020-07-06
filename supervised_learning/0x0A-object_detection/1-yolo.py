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
        image_height = image_size[0]
        image_width = image_size[1]
        input_shape = self.model.layers[0].input_shape
        for b, output in enumerate(outputs):
            anchors = self.anchors[b]
            grid_height, grid_width = output.shape[:2]
            anchor_boxes = output.shape[2]
            ph, pw = anchors.shape
            class_conf = 1 / (1 + np.exp(output[:, :, :, 4]))
            class_conf = class_conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box = output[..., :4]
            x = 1 / (1 + np.exp(box[..., 0])) + np.arange(grid_width).reshape((1, grid_width, 1)) / grid_width
            y = 1 / (1 + np.exp(box[..., 1])) + np.arange(grid_height).reshape((grid_height, 1, 1)) / grid_height
            w = pw * np.exp(box[..., 2]) / input_shape[1]
            h = ph * np.exp(box[..., 3]) / input_shape[2]
            classes = 1 / (1 + np.exp(output[:][:][:][5:]))
            box = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            box = np.moveaxis(box, 0, -1)
            boxes.append(box)
            box_confidence.append(class_conf)
            box_class_probs.append(classes)

        return boxes, box_confidence, box_class_probs
