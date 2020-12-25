import os
import cv2
import dlib
import json
import time
import torch
import pickle
import numpy as np
import datetime as dt
import tensorflow as tf
from torch import nn
from keras import Model
from torchvision import transforms
from keras.models import model_from_json
from constants import *


# модель эмоций
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return self.EMOTIONS_LIST[np.argmax(self.preds)]


# модель глаз
class CNNY(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.batch3 = nn.BatchNorm2d(num_features=64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(inplace=True)

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch1(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = self.batch2(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv3(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv4(h)
        h = self.batch3(h)
        h = self.act(h)

        h = self.conv5(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)

        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc3(h)
        return h


# модель улыбок
class CNNS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            padding=1)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1)

        self.batch1 = nn.BatchNorm2d(num_features=16)
        self.batch2 = nn.BatchNorm2d(num_features=32)
        self.batch3 = nn.BatchNorm2d(num_features=64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(inplace=True)

        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.batch1(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv2(h)
        h = self.batch2(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv3(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = self.conv4(h)
        h = self.batch3(h)
        h = self.act(h)

        h = self.conv5(h)
        h = self.batch3(h)
        h = self.act(h)
        h = self.pool(h)

        h = h.view(h.size(0), -1)

        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.fc3(h)
        return h


def analyse_video(video_path, timesstems):
    # агрузка моделей и отправка их на цп
    model_eye = torch.load(os.environ['PATH_TO_EYE_MODEL']).cpu()
    model_smile = torch.load(os.environ['PATH_TO_SMILE_MODEL']).cpu()
    with tf.device('/cpu:0'):
        model = FacialExpressionModel(os.environ['PATH_TO_JSON_MODEL'], os.environ['PATH_TO_WEIGHTS_MODEL'])
        new_model = Model(model.loaded_model.input, model.loaded_model.get_layer('dense_3').output)
        new_model.compile()

    model_eye.eval()
    model_smile.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

    # получение границ сегмента
    def get_key(dict):
        keys = []
        for key in dict.keys():
            keys.append(key)
        return keys

    j_key = get_key(timesstems)
    param_key = get_key(timesstems['0'])

    # чтение видео, путь так же надо из бд брать
    cap = cv2.VideoCapture(video_path)
    # создание детектора
    detector = dlib.get_frontal_face_detector()
    fps = cap.get(cv2.CAP_PROP_FPS)

    OUT = {}

    chunk_count = 0
    t0 = time.time()
    for i in j_key:
        frame_count = 0
        # получение границ и проверка на длинну
        frame_start = int(timesstems[str(i)][param_key[0]] / 100 * fps)
        frame_end = int(timesstems[str(i)][param_key[1]] / 100 * fps)
        print(f'CURRENT SEGMENT: {chunk_count}')
        if frame_end - frame_start < 30 * fps:
            print(f'PASSED SEGMENT {i} DUE TO < 30s')
            chunk_count += 1
        # основной цик где на кадре находиться хрюкальник и подаеться в модели
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

            # Обрати внимание, что эти показатели надо считывать для интервала
            # Соответственно, надо при каждом новом интервале их обновлять!!!!!
            trackers = []
            centrs = []
            eye_predict = []
            smile_predict = []
            emot_predict = []

            for frame_idx in range(frame_start, frame_end):

                ret, frame = cap.read()
                if not ret:
                    break
                print(f'{frame_count}/{frame_end - frame_start}')
                if frame_count == 0:
                    bboxes = detector(frame)

                    for i in bboxes:
                        x1 = i.left()
                        y1 = i.top()
                        x2 = i.right()
                        y2 = i.bottom()
                        (startX, startY, endX, endY) = (x1, y1, x2, y2)
                        tracker = dlib.correlation_tracker()

                        rect = dlib.rectangle(x1, y1, x2, y2)
                        tracker.start_track(frame, rect)
                        trackers.append(tracker)
                else:
                    for tracker in trackers:
                        tracker.update(frame)
                        pos = tracker.get_position()

                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())
                frame_count += 1

                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                centrs.append((cX, cY))

                face = frame[startY:endY, startX:endX]

                try:
                    face_r1 = cv2.resize(face, (64, 64))
                    face_r2 = cv2.resize(face, (48, 48))
                except cv2.error:
                    eye_predict.append(eye_predict[-1])
                    smile_predict.append(smile_predict[-1])
                    emot_predict.append(emot_predict[-1])
                    OUT["interval_" + str(chunk_count)] = {"eye": eye_predict,
                                                           "smile": smile_predict,
                                                           "emot": emot_predict,
                                                           "head_position": centrs}
                    print(f'AN ERROR OCCURED ON FRAME {frame_count}!!! WROTE PARAMETERS EQUAL OF THOSE ON PREVIOUS '
                          f'FRAME!!!')
                    continue
                face_y = transform(face_r1)
                face_y = torch.unsqueeze(face_y, 0)

                face_s = transform(face_r1)
                face_s = torch.unsqueeze(face_s, 0)

                face_g = cv2.cvtColor(face_r2, cv2.COLOR_RGB2GRAY)
                pred = new_model.predict(face_g[np.newaxis, :, :, np.newaxis])
                pred = pred[:, [0, 3, 6, 4, 2]]
                pred = np.exp(pred) / np.exp(pred).sum()

                with torch.no_grad():
                    result_eye = model_eye(face_y)
                    result_smile = model_smile(face_s)

                    eye_predict.append(result_eye)
                    smile_predict.append(result_smile)
                    emot_predict.append(pred)
                OUT["interval_" + str(chunk_count)] = {"eye": eye_predict,
                                                       "smile": smile_predict,
                                                       "emot": emot_predict,
                                                       "head_position": centrs}

            chunk_count += 1
    OUT["fps"] = fps
    return OUT
