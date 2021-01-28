import os
import cv2
import pickle
import json
import numpy as np


# def apply_motion_blur(image, size, angle):
#     k = np.zeros((size, size), dtype=np.float32)
#     k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
#     k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )
#     k = k * ( 1.0 / np.sum(k) )
#     return cv2.filter2D(image, -1, k)

# img = cv2.imread("1.jpg")

# blur_0 = apply_motion_blur(img, 10, 0)
# blur_1 = apply_motion_blur(img, 50, -120)
# blur_2 = apply_motion_blur(img, 10, 120)
# blur_3 = apply_motion_blur(img, 50, 120)

# bl01 = np.hstack([blur_0, blur_1])
# bl23 = np.hstack([blur_2, blur_3])
# bl = np.vstack([bl01, bl23])
# cv2.imshow("orig", img)
# cv2.imshow("blur", bl)
# cv2.waitKey(0)

#
# out = {
#     "1": [1, 0, 1],
#     "2": [0, 1, 0]
# }
# name = "test.data"
# path = "D:\Python"
# with open(os.path.join(path, name), "wb") as f:
#     pickle.dump(os.path.join(path, name), f)
# with open("timestemps.json", "r") as read_file:
#     data = json.load(read_file)
#
# cap = cv2.VideoCapture("video.mp4")
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# def get_key(dict):
#     keys = []
#     for key in dict.keys():
#         keys.append(key)
#     return keys
#
# j_key = get_key(data)
# param_key = get_key(data["0"])
#
# for i in j_key:
#     frame_start = int(data[str(i)][param_key[0]] / 1000 * fps)
#     frame_end = int(data[str(i)][param_key[1]] / 1000 *  fps)
#     print((frame_start, frame_end))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
#
#     for frame_idx in range(frame_start, frame_end):
#         ret, frame =cap.read()
#         if frame is not None:
#           print("yes")
#
#         cv2.imshow("frame", frame)
#         cv2.waitKey(1)
# cap.release()
# a = np.arange(12)
# b = np.array_split(a, 4)
# print(b)
# for i in b:
#     print(np.mean(i))

# a = [1, 2, 3]
# b = [4, 5, 6]
# c = [7, 8, 9]
# d = [a, b, c]
# mid = []
# for i in d:
#     mn.append(np.mean(i))
# print(mn)

# with open("audio_analysis.data", "rb") as f:
#     audio = pickle.load(f)
#
# for i in range(0, len(audio)):
#     for au in audio[i]["chunks"]:
#         print(au)

# import cv2  # Import the OpenCV library
# import numpy as np  # Import Numpy library
#
#
# # Project: Object Tracking
# # Author: Addison Sears-Collins
# # Website: https://automaticaddison.com
# # Date created: 06/13/2020
# # Python version: 3.7
#
# def main():
#     """
#     Main method of the program.
#     """
#
#     # Create a VideoCapture object
#     cap = cv2.VideoCapture(0)
#
#     # Create the background subtractor object
#     # Use the last 700 video frames to build the background
#     back_sub = cv2.createBackgroundSubtractorMOG2(history=700,
#                                                   varThreshold=25, detectShadows=True)
#
#     # Create kernel for morphological operation
#     # You can tweak the dimensions of the kernel
#     # e.g. instead of 20,20 you can try 30,30.
#     kernel = np.ones((20, 20), np.uint8)
#
#     while (True):
#
#         # Capture frame-by-frame
#         # This method returns True/False as well
#         # as the video frame.
#         ret, frame = cap.read()
#
#         # Use every frame to calculate the foreground mask and update
#         # the background
#         fg_mask = back_sub.apply(frame)
#
#         # Close dark gaps in foreground object using closing
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
#
#         # Remove salt and pepper noise with a median filter
#         fg_mask = cv2.medianBlur(fg_mask, 5)
#
#         # Threshold the image to make it either black or white
#         _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
#
#         # Find the index of the largest contour and draw bounding box
#         fg_mask_bb = fg_mask
#         contours, hierarchy = cv2.findContours(fg_mask_bb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#         areas = [cv2.contourArea(c) for c in contours]
#
#         # If there are no countours
#         if len(areas) < 1:
#
#             # Display the resulting frame
#             cv2.imshow('frame', frame)
#
#             # If "q" is pressed on the keyboard,
#             # exit this loop
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             # Go to the top of the while loop
#             continue
#
#         else:
#             # Find the largest moving object in the image
#             max_index = np.argmax(areas)
#
#         # Draw the bounding box
#         cnt = contours[max_index]
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#
#         # Draw circle in the center of the bounding box
#         x2 = x + int(w / 2)
#         y2 = y + int(h / 2)
#         cv2.circle(frame, (x2, y2), 4, (0, 255, 0), -1)
#
#         # Print the centroid coordinates (we'll use the center of the
#         # bounding box) on the image
#         text = "x: " + str(x2) + ", y: " + str(y2)
#         cv2.putText(frame, text, (x2 - 10, y2 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         # Display the resulting frame
#         cv2.imshow('frame', frame)
#
#         # If "q" is pressed on the keyboard,
#         # exit this loop
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
#     # Close down the video stream
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     main()