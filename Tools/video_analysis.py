import cv2
import dlib
import json
import time
import torch
from torchvision import transforms
from Models import *

model_eye = torch.load("F:/Python/Models/pth/eyeB_91.8.pth").cpu()
model_smile = torch.load("F:/Python/Models/pth/smileB_90.8.pth").cpu()
model_emot = torch.load("F:/Python/Models/pth/emotB_57.9.pth").cpu()

model_eye.eval()
model_smile.eval()
model_emot.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

get_label_eye = {0: "closed", 1: "open"}
get_label_smile = {0: "poker", 1: "smile"}
get_label_emot = {0: "angry",
                  1: "happy",
                  2: "neutral",
                  3: "sad",
                  4: "fear"}

with open("F:/Python/Data/Demo/timestemps_cor.json", "r") as read_file:
    timesstems = json.load(read_file)


def get_key(dict):
    keys = []
    for key in dict.keys():
        keys.append(key)
    return keys


j_key = get_key(timesstems)
param_key = get_key(timesstems["0"])

cap = cv2.VideoCapture("F:/Python/Data/Demo/video.mp4")
detector = dlib.get_frontal_face_detector()
fps = cap.get(cv2.CAP_PROP_FPS)

OUT = {}

chunk_count = 0
t0 = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
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

        trackers = []
        centrs = []
        eye_predict = []
        smile_predict = []
        emot_predict = []

        print(chunk_count)
        for frame_idx in range(frame_start, frame_end):

            ret, frame = cap.read()
            if not ret:
                break

            if face_detect == False:
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

            face_detect = True
            # print(startX)
            # print(startY)
            # print(endX)
            # print(endY)

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            centrs.append((cX, cY))

            face = frame[startY:endY, startX:endX]
            face_r = cv2.resize(face, (64, 64))
            # try:
            #     face_r = cv2.resize(face, (64, 64))
            # except cv2.error:
            #     eye_predict.append(result_eye[-1])
            #     smile_predict.append(result_smile[-1])
            #     emot_predict.append(result_emot[-1])
            #     continue

            face_y = transform(face_r)
            face_s = transform(face_r)
            face_e = transform(face_r)

            face_y = torch.unsqueeze(face_y, 0)
            face_s = torch.unsqueeze(face_s, 0)
            face_e = torch.unsqueeze(face_e, 0)

            with torch.no_grad():
                result_eye = model_eye(face_y)
                result_smile = model_smile(face_s)
                result_emot = model_emot(face_e)

                eye_predict.append(result_eye)
                smile_predict.append(result_smile)
                emot_predict.append(result_emot)

        chunk_count += 1

t1 = time.time()
print(t1 - t0)
cap.release()
cv2.destroyAllWindows()
