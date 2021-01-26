import os
import cv2
import random
import pickle
import numpy as np
from PIL import Image


# Кастомный моушен блюр
class MotionBlur(object):
    def __call__(self, image):
        self.size = random.randint(1, 50)
        self.angle = random.randint(-90, 90)
        self.k = np.zeros((self.size, self.size), dtype=np.float32)
        self.k[(self.size - 1) // 2, :] = np.ones(self.size, dtype=np.float32)
        self.k = cv2.warpAffine(self.k, cv2.getRotationMatrix2D((self.size / 2 - 0.5,
                                                                 self.size / 2 - 0.5),
                                                                self.angle, 1.0), (self.size, self.size))
        self.k = self.k * (1.0 / np.sum(self.k))

        img = np.array(image)
        img = cv2.filter2D(img, -1, self.k)
        return Image.fromarray(img.astype(np.uint8))


def save_list(name, list):
    with open(name + ".data", "wb") as f:
        pickle.dump(list, f)


def remove(path, ext1, ext2):
    dir = os.listdir(path)
    for i in dir:
        if i.endswith(ext1) or i.endswith(ext2):
            os.remove(os.path.join(path, i))
