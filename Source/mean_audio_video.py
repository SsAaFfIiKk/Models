import os
import json
import numpy as np

folder_save = "F:/Python/Models/Source/emot/mean"
path_to_audio = "F:/Python/Models/Source/emot/audio"
path_to_video = "F:/Python/Models/Source/emot/video"

audio_list = [os.path.join(path_to_audio, name) for name in os.listdir(path_to_audio) if name.endswith(".json")]
video_list = [os.path.join(path_to_video, name) for name in os.listdir(path_to_video) if name.endswith(".json")]
audio_key = "audio"
video_key = "video"

def get_logits_list(a_dict, a_key, v_dict, v_key):
    mean_emot = a_dict.copy()

    for i in range(len(a_dict["segment"]["chunks"])):
        a_logits = a_dict["segment"]["chunks"][i][a_key + "_emotion_model_logits"]
        v_logits = v_dict["segment"]["chunks"][i][v_key + "_emotion_model_logits"]
        del mean_emot["segment"]["chunks"][i][a_key + "_emotion_model_logits"]

        mean = mean_mass(a_logits, v_logits)
        mean_emot["segment"]["chunks"][i]["mean_emotion_model_logits"] = mean
    return mean_emot

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
    num = a_file.split("\\")[1].split("_")[0].split(".")[0]
    name_emot = num + "_mean_emotion_logits.json"

    with open(a_file, "rb") as af:
        audio_data = json.load(af)

    with open(v_file, "rb") as vf:
        video_data = json.load(vf)


    mean_dict = get_logits_list(audio_data, audio_key, video_data, video_key)

    with open(os.path.join(folder_save, name_emot), "w") as file:
        json.dump(mean_dict, file, indent=2)
