import torch
import librosa
import datetime
import pandas as pd
import numpy as np
from audio_tools import chunkizer, get_logits, load_model
from Models import EmoAlex
import warnings
warnings.filterwarnings("ignore")

start = 0
end = 3
step = 1

emotion_labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
}

log = {
    "Chunk_ind": [],
    "Predict": [],
    "Confidence": []
}

path_to_file = "F:/Python/Data/demo/audio_only_ivan.m4a"
path_to_weights = "F:/Python/Data/model_data/AlexNet.pt"
path_to_save = "F:/Python/Data/Audio/segments"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmoAlex()
model = load_model(model, path_to_weights, device)

print("Load audio")
print(datetime.datetime.now())
audio, _ = librosa.load(path_to_file, 48000)

print("Start spliting")
print(datetime.datetime.now())
chunks = chunkizer(audio, 1)

for ind in range(len(chunks)):

    chunk = np.hstack(chunks[start:end])
    logit = get_logits(chunk, model, device)

    class_number = np.argmax(logit)
    log["Confidence"].append(round(logit[class_number], ndigits=3))

    class_name = emotion_labels[class_number]

    log["Chunk_ind"].append(ind)
    log["Predict"].append(class_name)
    start += step
    end += step

df = pd.DataFrame(data=log)
print("Save logs")
print(datetime.datetime.now())
# df.to_csv("F:/Python/Data/Audio/logv2.csv", sep=",", index=False)
print("Done")
