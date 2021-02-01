import cv2
import numpy as np

#в этом файле храниться описание модели
prototxt_path = "F:/Python/Data/model_data/deploy.prototxt"
# в этом веса
caffemodel_path = "F:/Python/Data/model_data/weights.caffemodel"
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    (h, w) = frame.shape[:2]
    # псоздания блоба и переда его в нейронку
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    # обход всех найденых лиц
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)

        if confidence > 0.5:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 1, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
