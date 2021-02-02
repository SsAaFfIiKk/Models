import cv2
import json
import time
import torch
import pickle
import numpy as np
from torchvision import transforms
from Models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_eye = torch.load("F:/Python/Models/pth/eyeB_91.8.pth", map_location=torch.device(device))
model_smile = torch.load("F:/Python/Models/pth/smileB_90.8.pth", map_location=torch.device(device))
model_emot = torch.load("F:/Python/Models/pth/emotB_57.9.pth", map_location=torch.device(device))

model_eye.eval()
model_smile.eval()
model_emot.eval()

prototxt_path = "F:/Python/Data/model_data/deploy.prototxt"
caffemodel_path = "F:/Python/Data/model_data/weights.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

get_label_emot = {0: "angry",
                  1: "happy",
                  2: "neutral",
                  3: "sad",
                  4: "fear"}

with open("F:/Python/Data/demo/timestemps_cor.json", "r") as read_file:
    timesstems = json.load(read_file)


def get_key(dict):
    keys = []
    for key in dict.keys():
        keys.append(key)
    return keys


def detect(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        coords = box.astype("int")

        return coords


j_key = get_key(timesstems)
param_key = get_key(timesstems["0"])

cap = cv2.VideoCapture("F:/Python/Data/Demo/video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

chunk_count = 0
t0 = time.time()
font = cv2.FONT_HERSHEY_TRIPLEX

for i in j_key:
    face_detect = False

    frame_start = int(timesstems[str(i)][param_key[0]] / 100 * fps)
    frame_end = int(timesstems[str(i)][param_key[1]] / 100 * fps)

    if frame_end - frame_start < 30 * fps:
        chunk_count += 1
        print("Skip " + str(i))
        pass

    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        log = {}
        predicts = []
        print("Current chunk: ", chunk_count)
        log["segment_" + str(chunk_count)] = chunk_count
        for frame_idx in range(frame_start, frame_end):

            ret, frame = cap.read()
            coords = detect(frame)

            cX = int((coords[0] + coords[2]) / 2.0)
            cY = int((coords[1] + coords[3]) / 2.0)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (255, 255, 255), 2)
            cv2.circle(frame, (cX, cY), 1, (255, 255, 255), 2)

            face = frame[coords[0]:coords[2], coords[1]:coords[3]]
            face_r = cv2.resize(face, (64, 64))

            face_t = transform(face_r)
            face_u = torch.unsqueeze(face_t, 0)
            face_u = face_u.to(device)

            with torch.no_grad():
                result_eye = model_eye(face_u)
                result_smile = model_smile(face_u)
                result_emot = model_emot(face_u)

                label_eye = result_eye.argmax(dim=1)
                label_smile = result_smile.argmax(dim=1)
                label_emot = result_emot.argmax(dim=1)

                anser_emot = get_label_emot[label_emot.sum().item()]

                predicts.append({'head_pos': (cX, cY),
                                  'emo_class': anser_emot,
                                  'blink': label_eye.sum().item(),
                                  'smile': label_smile.sum().item()
                                  })
            cv2.putText(frame, "eye: " + str(label_eye.sum().item()), (15, 240), font, 1, (0, 255, 0), 1)
            cv2.putText(frame, "smile: " + str(label_smile.sum().item()), (15, 270), font, 1, (0, 255, 0), 1)
            cv2.putText(frame, "emotion:" + anser_emot , (15, 300), font, 1, (0, 255, 0), 1)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        log["frames"] = predicts
        with open("chunk" + str(chunk_count) + ".data", "wb") as f:
            pickle.dump(log, f)
        chunk_count += 1

t1 = time.time()
print(t1 - t0)
cap.release()
cv2.destroyAllWindows()
