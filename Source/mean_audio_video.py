import os
import json
import numpy as np

folder_save = "F:/Python/Models/Source/emot/mean"
path_to_audio = "F:/Python/Models/Source/emot/audio"
path_to_video = "F:/Python/Models/Source/emot/video"

audio_list = [os.path.join(path_to_audio, name) for name in os.listdir(path_to_audio) if name.endswith(".json")]
video_list = [os.path.join(path_to_video, name) for name in os.listdir(path_to_video) if name.endswith(".json")]


def get_logits_list(dict, key):
    logits = []
    for i in range(len(dict["segment"]["chunks"])):
        logits.append(dict["segment"]["chunks"][i][key + "_emotion_model_logits"])
    return logits


def mean_mass(fst, scd):
    mean_mas = []
    for ls_a, ls_v in zip(fst, scd):
        mean_mas.append(((np.array(ls_a) + np.array(ls_v)) / 2.0).reshape(-1).tolist())

    return mean_mas

def get_mean(a_logits, v_logits):
    for al, vl in zip(a_logits, v_logits):
        print(len(al), len(vl))
        return mean_mass(al, vl)


for a_file, v_file in zip(audio_list, video_list):

    with open(a_file, "rb") as af:
        audio_data = json.load(af)

    with open(v_file, "rb") as vf:
        video_data = json.load(vf)

    a_logit = get_logits_list(audio_data, "audio")
    v_logit = get_logits_list(video_data, "video")

    mean = get_mean(a_logit, v_logit)
