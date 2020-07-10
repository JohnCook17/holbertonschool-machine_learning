#!/usr/bin/env python3
"""You only look once"""
import numpy as np
import tensorflow as tf
import os
import cv2


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
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        boxes = []
        box_confidence = []
        box_class_probs = []
        image_height = image_size[0]
        image_width = image_size[1]
        input_shape = self.model.input.shape
        for b, output in enumerate(outputs):
            anchors = self.anchors[b]
            grid_height, grid_width = output.shape[:2]
            anchor_boxes = output.shape[2]
            pw = anchors[:, 0]
            ph = anchors[:, 1]
            pw = pw.reshape(1, 1, len(pw))
            ph = ph.reshape(1, 1, len(ph))
            # set up boxes
            box = output[..., :4]
            x = sigmoid(box[..., 0])
            y = sigmoid(box[..., 1])
            cx = (np.tile(np.arange(grid_width), grid_height).reshape
                  (grid_width, grid_width, 1))
            cy = (np.tile(np.arange(grid_width), grid_height).reshape
                  (grid_height, grid_height).T.reshape(grid_height,
                                                       grid_height, 1))
            # cx = np.arange(grid_width).reshape((1, grid_width, 1))
            # cy = np.arange(grid_height).reshape((grid_height, 1, 1))
            x = x + cx
            y = y + cy
            x = x / grid_width
            y = y / grid_height
            w = (pw * np.exp(box[..., 2]))
            h = (ph * np.exp(box[..., 3]))
            w = w / int(input_shape[1])
            h = h / int(input_shape[2])
            box = np.array([(x - w / 2) * image_width,
                            (y - h / 2) * image_height,
                            (x + w / 2) * image_width,
                            (y + h / 2) * image_height])
            box = np.moveaxis(box, 0, -1)
            # set up class conf
            class_conf = 1 / (1 + np.exp(-output[:, :, :, 4]))
            class_conf = class_conf.reshape(grid_height,
                                            grid_width,
                                            anchor_boxes, 1)
            # set up classes
            classes = 1 / (1 + np.exp(-output[:, :, :, 5:]))
            # append to list
            boxes.append(box)
            box_confidence.append(class_conf)
            box_class_probs.append(classes)

        return boxes, box_confidence, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filters the boxes, if no class then it is removed"""
        filtered_boxes_l = []
        box_classes_l = []
        box_class_scores_l = []
        threshold = self.class_t
        for index in range(len(box_class_probs)):
            classes = box_class_probs[index].shape[-1]
            box_scores = box_confidences[index] * box_class_probs[index]
            box_class_scores = np.argmax(box_scores, axis=-1)
            box_scores = np.max(box_scores, axis=-1)
            idx = np.where(box_scores >= threshold)
            filtered_boxes_l.append(boxes[index][idx])
            box_classes_l.append(box_class_scores[idx])
            box_class_scores_l.append(box_scores[idx])
            filtered_boxes_r = np.concatenate(filtered_boxes_l, axis=0)
            box_classes_r = np.concatenate(box_classes_l, axis=0)
            box_class_scores_r = np.concatenate(box_class_scores_l, axis=0)
        return (filtered_boxes_r,
                box_classes_r,
                box_class_scores_r)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Removes duplicate boxes"""
        pred_boxes_l = []
        pred_scores_l = []
        pred_classes_l = []
        pred = np.unique(box_classes)
        for c in pred:
            ind = np.where(box_classes == c)
            b = filtered_boxes[ind]
            bs = box_scores[ind]
            bcl = box_classes[ind]
            # want a sorted index of scores
            sorted_ind = np.flip(np.argsort(bs))
            # keep the ind we want
            keep_ind = []
            while sorted_ind.size > 1:
                # keep ind with thresh < iou remove highest, keep remaining
                maximum = sorted_ind[0]
                others = sorted_ind[1:]
                keep_ind.append(maximum)
                bmax = filtered_boxes[maximum]
                bother = filtered_boxes[others]
                x1 = np.maximum(bmax[0], bother[:, 0])
                y1 = np.maximum(bmax[1], bother[:, 1])
                x2 = np.minimum(bmax[2], bother[:, 2])
                y2 = np.minimum(bmax[3], bother[:, 3])

                intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
                Amax = (bmax[2] - bmax[0]) * (bmax[3] - bmax[1])
                Aother = ((bother[:, 2] - bother[:, 0]) *
                          (bother[:, 3] - bother[:, 1]))

                iou = intersection / (Amax + Aother - intersection)
                below_ind = np.where(iou < self.nms_t)
                # not sure why below_ind is a tuple :(
                sorted_ind = sorted_ind[below_ind[0] + 1]

            if sorted_ind.size == 1:
                keep_ind.append(sorted_ind[0])

            keep_ind = np.array(keep_ind)
            pred_boxes_l.append(b[keep_ind])
            pred_classes_l.append(bcl[keep_ind])
            pred_scores_l.append(bs[keep_ind])
            pred_boxes = np.concatenate(pred_boxes_l, axis=0)
            pred_classes = np.concatenate(pred_classes_l, axis=0)
            pred_scores = np.concatenate(pred_scores_l, axis=0)
        return (pred_boxes,
                pred_classes,
                pred_scores)

    @staticmethod
    def load_images(folder_path):
        """"""
        images = []
        img_paths = []
        for file_name in os.listdir(folder_path):
            img = cv2.imread(folder_path + "/" + file_name)
            if img is not None:
                images.append(img)
                img_paths.append(folder_path + "/" + file_name)
        return (images, img_paths)

    def preprocess_images(self, images):
        """preprocess the images"""
        pimages = []
        image_shapes = []
        shape = self.model.input.shape
        shape = (shape[1], shape[2])
        for image in images:
            image_height, image_width = image.shape[0], image.shape[1]
            image = cv2.resize(image,
                               dsize=shape,
                               interpolation=cv2.INTER_CUBIC)
            image_shape = [image_height, image_width]
            image_shapes.append(image_shape)
            image = image / 255
            pimages.append(image)
        return np.asarray(pimages), np.asarray(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """"""
        class_list = []
        for j in range(len(boxes)):
            class_str = ""
            label = - 1
            for i in range(len(box_classes)):
                class_str = str(self.class_names[box_classes[i]])
                label = i
                class_list.append(class_str)
            if label >= 0:
                print(boxes[j])
                x1, y1, x2, y2 = boxes[j]
                start = (x1, y1)
                end = (x2, y2)
                cv2.rectangle(img=image,
                              pt1=(int(x1), int(y1)),
                              pt2=(int(x2), int(y2)),
                              color=(255, 0, 0),
                              thickness=2)
                cv2.putText(img=image,
                            text=(class_list[j] + " " +
                                  str(round(box_scores[j] * 100))),
                            org=(int(x1), int(y1) - 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA
                            )
        cv2.imshow(file_name, image)
        if cv2.waitKey(0) == ord("s"):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            cv2.imwrite("detections/" + file_name, image)
        cv2.destroyAllWindows()
