import cv2
import dlib
import time



frame_count = 0
trackers = []
cap = cv2.VideoCapture("F:/Python/emotion.mp4")
detector = dlib.get_frontal_face_detector()
t0 = time.time()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    if frame_count == 0:
        bboxes = detector(frame)

        for i in bboxes:
            x1 = i.left()
            y1 = i.top()
            x2 = i.right()
            y2 = i.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
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

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

    frame_count += 1

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

t1 = time.time()
print(t1 - t0)
cv2.destroyAllWindows()
cap.release()
