import pickle
import numpy as np


def chunks(list, chunk_size):
    return [list[i:i+chunk_size] for i in range(0, len(list), chunk_size)]


def get_predict(list, get_label):
    list_pr = []
    for i in list:
        label = i.argmax(dim=1)
        anser = get_label[label.sum().item()]
        list_pr.append(anser)
    chunks_name = chunks(list_pr, 30)
    return chunks_name


def get_dict(emp_dict, list, fs_state, sd_state):
    for i, count in enumerate(list):
        fs = count.count(fs_state)
        sd = count.count(sd_state)
        emp_dict[i] = {fs_state: fs,
                       sd_state: sd}
    return emp_dict


def get_key(dict):
    keys = []
    for key in dict.keys():
        keys.append(key)
    return keys


with open("result.data", "rb") as f:
    results = pickle.load(f)

get_label_eye = {0: "closed", 1: "open"}
get_label_smile = {0: "poker", 1: "smile"}

update_result = {}
interval_key = get_key(results)
fps = results[interval_key[-1]]
mn = int(3 * fps)

chunk_key = get_key(results[interval_key[0]])
len_interval_key = len(interval_key) - 1

if __name__ == '__main__':
    for i in range(len_interval_key):
        eye_entry = {}
        smile_entry = {}
        emot_mean = []
        emot = results[interval_key[i]][chunk_key[2]]
        three_s = chunks(emot, mn)
        for count in three_s:
            emot_mean.append(np.mean(count, axis=0))

        eye = results[interval_key[i]][chunk_key[0]]
        smile = results[interval_key[i]][chunk_key[1]]

        eye_predict = get_predict(eye, get_label_eye)
        smile_predict = get_predict(smile, get_label_smile)

        eye_entry = get_dict(eye_entry, eye_predict, "open", "closed")
        smile_entry = get_dict(smile_entry, smile_predict, "smile", "poker")

        update_result[interval_key[i]] = {"emotion": emot_mean,
                                          "eye": eye_entry,
                                          "smile": smile_entry
                                          }
# print(update_result["interval_0"]["emotion"][0])
# print(update_result["interval_1"]["emotion"][0])
# print(update_result["interval_2"]["emotion"][0])
