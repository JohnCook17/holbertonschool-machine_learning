#!/usr/bin/env python3
""""""
import cv2
import dlib
import numpy as np


class FaceAlign:
    """"""
    def __init__(self, shape_predictor_path):
        """"""
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """"""
        rectangels = self.detector(image, 1)
        return_rectangle = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
        try:
            rectangel_area = 0
            for rectangel in rectangels:
                current_rect_area = rectangel.area()
                if current_rect_area > rectangel_area:
                    rectangel_area = current_rect_area
                    return_rectangle = rectangel
            return return_rectangle
        except Exception as e:
            print(e)
            print("detect failed")
            return None

    def find_landmarks(self, image, detection):
        """"""
        try:
            my_list = []
            parts = self.shape_predictor(image, detection).parts()
            for part in parts:
                my_list.append((part.x, part.y))
            return np.asarray(my_list)
        except Exception as e:
            print(e)
            print("find landmarks failed")
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """"""
        landmarks = self.find_landmarks(image, self.detect(image))
        if landmarks is None:
            return None
        landmarks = np.asarray([landmarks[landmark_indices[0]], landmarks[landmark_indices[1]], landmarks[landmark_indices[2]]]).astype(np.float32)
        anchor_points = anchor_points * size
        warp_mat = cv2.getAffineTransform(landmarks, anchor_points)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))
        return warp_dst
