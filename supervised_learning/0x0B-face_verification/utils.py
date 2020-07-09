#!/usr/bin/env python3
""""""
import numpy as np
import cv2
import os


def load_images(images_path, as_array=True):
    """"""
    images = []
    filenames = []
    for filename in os.listdir(images_path):
        img = cv2.imread(images_path + "/" + filename, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        filenames.append(filename)
    if as_array is True:
        images = np.asarray(images)
    return images, filenames


def load_csv(csv_path, params={}):
    """"""
    lines = []
    with open(csv_path) as f:
        line = f.readline().splitlines()
        while line:
            lines.append(line[0].split(","))
            line = f.readline().splitlines()
    return lines


def save_images(path, images, filenames):
    """"""
    for image, filename in zip(images, filenames):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            saved = cv2.imwrite(path + "/" + filename, image)
        except Exception as e:
            print(e)
            print("save images failed")
            print(filename)
            return False
    return saved


def generate_triplets(images, filenames, triplet_names):
    """"""
    images = images[:] / 255
    triplets = []
    filenames = np.asarray(filenames)
    for list_of_trip in triplet_names:
        for name in list_of_trip:
            index = np.where(filenames == name + ".jpg")
            triplets.append(np.asarray(images[index], dtype="float32"))
    triplets = np.concatenate(triplets, axis=0)
    triplets = [triplets[0:: 3], triplets[1:: 3], triplets[2:: 3]]
    return triplets
