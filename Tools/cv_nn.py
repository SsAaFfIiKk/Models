import cv2
import numpy as np

prototxt_path = "D:/Python/Create_Face_Data/model_data/deploy.prototxt"
caffemodel_path = "D:/Python/Create_Face_Data/model_data/weights.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
