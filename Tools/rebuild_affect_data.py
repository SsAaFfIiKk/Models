import os
import cv2
import csv

path_to_dataset = "F:/Affect"
csv_train_name = "training.csv"
csv_valid_name = "validation.csv"
path_to_save = "S:/Rebuild_affect"
training = os.path.join(path_to_dataset, csv_train_name)

emot_label = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"}

Neutral = 0
Happy = 0
Sad = 0
Surprise = 0
Fear = 0
Disgust = 0
Anger = 0
Contempt = 0

with open(training, 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        folder_name = row['subDirectory_filePath']
        expression = int(row['expression'][0:])
        if expression in emot_label:
            img = cv2.imread(os.path.join(path_to_dataset, folder_name))

            if expression == 0:
                cv2.imwrite(os.path.join(path_to_save, str(Neutral) + "_" + emot_label[expression]) + ".jpg", img)
                Neutral += 1
            if expression == 1:
                cv2.imwrite(os.path.join(path_to_save, str(Happy) + "_" + emot_label[expression]) + ".jpg", img)
                Happy += 1
            if expression == 2:
                cv2.imwrite(os.path.join(path_to_save, str(Sad) + "_" + emot_label[expression]) + ".jpg", img)
                Sad += 1
            if expression == 3:
                cv2.imwrite(os.path.join(path_to_save, str(Surprise) + "_" + emot_label[expression]) + ".jpg", img)
                Surprise += 1
            if expression == 4:
                cv2.imwrite(os.path.join(path_to_save, str(Fear) + "_" + emot_label[expression]) + ".jpg", img)
                Fear += 1
            if expression == 5:
                cv2.imwrite(os.path.join(path_to_save, str(Disgust) + "_" + emot_label[expression]) + ".jpg", img)
                Disgust += 1
            if expression == 6:
                cv2.imwrite(os.path.join(path_to_save, str(Anger) + "_" + emot_label[expression]) + ".jpg", img)
                Anger += 1
            if expression == 7:
                cv2.imwrite(os.path.join(path_to_save, str(Contempt) + "_" + emot_label[expression]) + ".jpg", img)
                Contempt += 1
print(Neutral)
print(Happy)
print(Sad)
print(Surprise)
print(Fear)
print(Disgust)
print(Anger)
print(Contempt)