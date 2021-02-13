import cv2
import torch
import numpy as np
from torchvision import transforms


class VideoAnalyze:
    def __init__(self, model_eye, model_smile, model_emot, model_cv):
        self.model_eye = model_eye
        self.model_smile = model_smile
        self.model_emot = model_emot
        self.model_cv = model_cv
        self.face_detect = False
        self.face = None

    def prepare_transforms(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])

    def get_face_borders(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model_cv.setInput(blob)
        detections = self.model_cv.forward()

        for found in range(0, detections.shape[2]):
            box = detections[0, 0, found, 3:7] * np.array([w, h, w, h])
            cor = box.astype("int")

            return cor

    def check_face(self, frame, cor):
        self.face = frame[cor[1]:cor[3], cor[0]:cor[2]]
        if len(self.face) != 0:
            self.face_detect = True
            self.prepare_face()
        else:
            self.face_detect = False

    def prepare_face(self):
        face_r = cv2.resize(self.face, (64, 64))
        face_t = self.transform(face_r)
        self.face_u = torch.unsqueeze(face_t, 0)

    def get_face_position(self, cor):
        cX = int((cor[0] + cor[2]) / 2.0)
        cY = int((cor[1] + cor[3]) / 2.0)
        return (cX, cY)

    def get_eye_predict(self):
        result_eye = self.model_eye(self.face_u)
        label_eye = result_eye.argmax(dim=1)
        return label_eye.sum().item()

    def get_smile_predict(self):
        result_smile = self.model_smile(self.face_u)
        label_smile = result_smile.argmax(dim=1)
        return label_smile.sum().item()

    def get_emot_predict(self):
        result_emot = self.model_emot(self.face_u)
        return result_emot.detach().numpy()
