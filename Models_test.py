import cv2
import dlib
import torch
import time
from torch import nn
from torchvision import transforms


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


class CNN(nn.Module):
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

        self.fc1 = nn.Linear(in_features=1024, out_features=256,)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=5)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model_eye = torch.load("pth/eyeB_91.8.pth", map_location=torch.device(device))
# model_smile = torch.load("pth/smileB_90.8.pth", map_location=torch.device(device))
# model_emot = torch.load("pth/emotB_57.9.pth", map_location=torch.device(device))
res18 = torch.load("pth/res18_emot_67.4.pth", map_location=torch.device(device))
res50 = torch.load("pth/res50_emot_67.5.pth", map_location=torch.device(device))

# model_eye.eval()
# model_smile.eval()
# model_emot.eval()
res18.eval()
res50.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

# get_label_eye = {0: "closed", 1: "open"}
# get_label_smile = {0: "poker", 1: "smile"}
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
        # result_eye = model_eye(face_u)
        # result_smile = model_smile(face_u)
        # result_emot = model_emot(face_u)
        res18_result = res18(face_u)
        res50_result = res50(face_u)

        # label_eye = result_eye.argmax(dim=1)
        # label_smile = result_smile.argmax(dim=1)
        # label_emot = result_emot.argmax(dim=1)
        label_res18 = res18_result.argmax(dim=1)
        label_res50 = res50_result.argmax(dim=1)

        # anser_eye = get_label_eye[label_eye.sum().item()]
        # anser_smile = get_label_smile[label_smile.sum().item()]
        # anser_emot = get_label_emot[label_emot.sum().item()]
        anser18 = get_label_emot[label_res18.sum().item()]
        anser50 = get_label_emot[label_res50.sum().item()]

        # cv2.putText(frame, anser_eye, (25, 20), font, 1, (255, 0, 0), 1)
        # cv2.putText(frame, anser_smile, (25, 40), font, 1, (255, 0, 0), 1)
        # cv2.putText(frame, anser_emot, (25, 60), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, "res18: " + anser18, (25, 20), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, "res50: " + anser50, (25, 50), font, 1, (255, 0, 0), 1)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

t1 = time.time()
print(t1 - t0)
