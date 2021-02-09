import cv2
import dlib
import torch
import time
from tools import load_model
from torchvision import transforms
from Source.Models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

path_to_eye_model = "pth/eyeB_91.8.pth"
path_to_smile_model = "pth/smileB_90.8.pth"
path_to_emot_model = "pth/emotB_57.9.pth"

model_eye = load_model(path_to_eye_model, device)
model_smile = load_model(path_to_smile_model, device)
model_emot = load_model(path_to_emot_model, device)

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

cap = cv2.VideoCapture("F:/Python/Data/emotion.mp4")
# cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

t0 = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
Face_detect = False

trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if Face_detect == False:
        bboxes = detector(frame)

        for i in bboxes:
            x1 = i.left()
            y1 = i.top()
            x2 = i.right()
            y2 = i.bottom()
            (startX, startY, endX, endY) = (x1, y1, x2, y2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
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
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

    Face_detect = True

    face = frame[startY:endY, startX:endX]
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


        anser_eye = get_label_eye[label_eye.sum().item()]
        anser_smile = get_label_smile[label_smile.sum().item()]
        anser_emot = get_label_emot[label_emot.sum().item()]

        cv2.putText(frame, anser_eye, (25, 20), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, anser_smile, (25, 40), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, anser_emot, (25, 60), font, 1, (255, 0, 0), 1)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

t1 = time.time()
print(t1 - t0)
