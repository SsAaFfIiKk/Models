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
with open("audio_analysis.data", "rb") as f:
    audio = pickle.load(f)

for i in range(0, len(audio)):
    for au in audio[i]["chunks"]:
        print(au)
