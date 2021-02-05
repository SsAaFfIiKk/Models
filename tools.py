import os
import cv2
import random
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms


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


def save_list(name, lst):
    with open(name + ".data", "wb") as f:
        pickle.dump(lst, f)


def remove(path, *exts):
    dr = os.listdir(path)
    for i in dr:
        if i.split(".")[-1] in exts:
            os.remove(os.path.join(path, i))

# def remove(path, ext1, ext2):
#     dr = os.listdir(path)
#     for i in dr:
#         if i.endswith(ext1) or i.endswith(ext2):
#             os.remove(os.path.join(path, i))

def transform(r_size):
    test_transform = transforms.Compose([
        transforms.Resize(r_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    train_transform = transforms.Compose([
        transforms.Resize(r_size),
        # Случайное применение гаусофского или моушен блюра
        transforms.RandomApply([MotionBlur(),
                                transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0))],
                               p=0.3),
        # Изменение перспективы, параметры взяты из документации
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0),
        transforms.RandomCrop(64, padding=6),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    return test_transform, train_transform
