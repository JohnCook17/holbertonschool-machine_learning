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
        # print(self.anchors.shape)
        for b, output in enumerate(outputs):
            anchors = self.anchors[b]
            # print(anchors.shape)
            # anchor_number = output.shape[2]
            # print(anchors)
            grid_height, grid_width = output.shape[:2]
            # print(grid_height, grid_width)
            image_height = image_size[0]
            image_width = image_size[1]
            # print(image_height, image_width)
            # output = output.reshape((grid_height, grid_width, anchor_number, -1))
            # output[..., :2] = 1 / (1 + np.exp(-output[..., :2]))
            # output[..., 4:] = 1 / (1 + np.exp(-output[..., 4:]))
            ph, pw = anchors.shape
            # class_conf = output[:][:][b][4]
            print(output[..., b, :4].shape)
            box = output[..., b, :4]
            x = 1 / (1 + np.exp(box[..., 0]))
            y = 1 / (1 + np.exp(box[..., 1]))
            w = pw * np.exp(box[..., 2])
            h = ph * np.exp(box[..., 3])
            # print(x)
            # print(x - w / 2)
            # classes = output[:][:][b][5:]
            box = np.array([rx - rw / 2, ry - rh / 2, rx + rw / 2, ry + rh /2])
            boxes.append(box)
            # box_confidence.append(class_conf)
            # box_class_probs.append(classes)

        return boxes, None, None
