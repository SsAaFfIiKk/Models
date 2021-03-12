import os
import pandas as pd
import soundfile as sf
from audio_tools import *
from Models import EmoAlex


sr = 48000
emotion_labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Scared"
}

log = {
    "Path_to_file": [],
    "File_ind": [],
    "Predict": [],
    "Confidence": []
}

path_to_file = "F:/Python/Data/Audio/audio_only_ivan.m4a"
path_to_weights = "F:/Python/Data/model_data/AlexNet.pt"
path_to_save = "F:/Python/Data/Audio/segments"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmoAlex()
model = load_model(model, path_to_weights, device)

audio, _ = librosa.load(path_to_file, sr)
chunks = chunkizer(audio, 3)

for ind, chunk in enumerate(chunks):

    file_name = os.path.join(path_to_save, str(ind) + ".wav")
    sf.write(file_name, chunk, sr, "PCM_24")

    logit = get_logits(chunk, model, device)
    # print(emotion_labels[np.argmax(logit)])

    class_number = np.argmax(logit)
    if class_number == 4:
        class_number = 2
    class_name = emotion_labels[class_number]

    log["Path_to_file"].append(file_name)
    log["File_ind"].append(ind)
    log["Predict"].append(class_name)
    log["Confidence"].append(round(logit[class_number], ndigits=3))

df = pd.DataFrame(data=log)
df.to_csv("F:/Python/Data/Audio/log.csv", sep=",", index=False)
