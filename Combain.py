import pickle
import numpy as np
from scipy import stats


def get_chunks(lst, chunk_size):
    """
    разделяем список на 30 секунд
    если длянна меньше 30 секунд то пропускаем его
    """
    list_30s = []
    for idx_30s in range(0, len(lst), chunk_size):
        if len(lst[idx_30s:idx_30s+chunk_size]) == chunk_size:
            list_30s.append(lst[idx_30s:idx_30s+chunk_size])
    return list_30s


def mean_mass(fst, scd):
    return (np.array(fst) + np.array(scd)) / 2.0


def get_labels(predictions_list):
    '''
    Функция просто преобразовывает logits в номера меток
    '''
    labels_list = []
    for prediction in predictions_list:
        label = prediction.argmax(dim=1).item()
        labels_list.append(label)

    return labels_list


def compute_states_num(states):
    '''
    Функция вычисляет количество изменений во времени
    '''
    # счетчик состояний
    state_cnt = 0

    # предыдущее состояние нужно для распознавания перехода между состояниями
    prev_state = None

    for current_state in states:
        if prev_state is not None:
            # условие перехода из одного состояния в другое
            if prev_state == 1 and current_state == 0:
                state_cnt += 1

        prev_state = current_state

    # обработка случая, когда последний элемент в состоянии 1
    if states[-1] == 1:
        state_cnt += 1

    return state_cnt


def compute_states_on_30s(states_list, step):
    '''
    Функция вычисляет количество тех или иных состояний (морганий, улыбок)
    на 30с интервале времени
    '''
    step_counts_list = []
    for idx_30s_chunk in range(0, len(states_list), step):
        # если отрезок меньше 30 с, не считаем его
        if len(states_list[idx_30s_chunk:idx_30s_chunk + step]) == step:
            states_num = compute_states_num(states_list[idx_30s_chunk:idx_30s_chunk + step])
            step_counts_list.append(states_num)

    return step_counts_list


def get_head_movs_num(head_position, fps):
    '''
    Функция служит для вычисления количества движений головы на заданном временном интервале
    на основе вычисления отклонений от доверительного интервала на 1 секундных отрезках

    head_list - список координат центра локализационной рамки лица на каждом кадре
    fps - количество кадров в секунду. Надо для расчета количества движений на интервале в 1 секунду
    '''
    chunk_len = len(head_position)
    head_mov_num = 0
    for i in range(0, chunk_len, fps):
        # get head coordinates on 1-second period
        chunk = head_position[i:i+fps]
        # compute the average position of head
        avg_position = chunk.mean(axis=0)
        # compute the distances to the average head position
        dists = np.linalg.norm(chunk - avg_position, axis=1)
        # compute the confidence interval according to Sudents t-test
        conf = stats.t.interval(0.995, df=len(dists)-1, loc=dists.mean(), scale=stats.sem(dists))
        # filter distances that don't lay in the confidence iterval
        head_mov_num += (dists < conf[0]).sum() + (dists > conf[1]).sum()

    return head_mov_num


def get_key(dct):
    keys = []
    for key in dct.keys():
        keys.append(key)
    return keys


def emo_change_count(lst):
    neutral = 2
    emo_count = 0
    ind = []
    for id, vl in enumerate(lst):
        ind.append(vl.index(max(vl)))
    for i in ind:
        if ind[i-1] != neutral and ind[i+1] != neutral:
            emo_count += 1

    return  emo_count

if __name__ == '__main__':
    with open("result.data", "rb") as f:
        results = pickle.load(f)

    with open("audio_analysis.data", "rb") as f:
        audio = pickle.load(f)

    interval_key = get_key(results)
    audio_key = get_key(audio[0]['chunks'][0])

    fps = int(results[interval_key[-1]])
    step = 30 * fps
    interval_key_len = len(interval_key) - 1

    for i in range(interval_key_len):
        emot_mean = []
        head_mov_30s_list = []
        mean_mas = []

        # получаеть все предикты для сегментв
        emot = results[interval_key[i]]["emot"]
        # делим на чанки
        thirty_s = get_chunks(emot, 30 * fps)
        # усредняем их
        for count in thirty_s:
            three_s = get_chunks(count, 3 * fps)
            for pr in three_s:
                emot_mean.append(np.mean(pr, axis=0).reshape(-1).tolist())
        emo_change = emo_change_count(emot_mean)

        # получаем предикты для глаз
        eye = results[interval_key[i]]["eye"]
        # конвертируем в метки
        eye_labels_list = get_labels(eye)
        # получаем количество на каждом 30-секундном интервале
        eye_predict = compute_states_on_30s(eye_labels_list, step)

        # получаем предикты для улыбок
        smile = results[interval_key[i]]["smile"]
        # конвертируем в метки
        smile_labels_list = get_labels(smile)
        # получаем количество на каждом 30-секундном интервале
        smile_predict = compute_states_on_30s(smile_labels_list, step)

        head_arr = np.array(results[interval_key[i]]['head_position'])
        # получаем количество на 30-секундном интервале
        for idx_30s_chunk in range(0, len(head_arr), step):
            # если отрезок меньше 30 с, не считаем его
            if len(head_arr[idx_30s_chunk:idx_30s_chunk+step]) == step:
                head_mov_num = get_head_movs_num(
                    head_arr[idx_30s_chunk:idx_30s_chunk+step], fps)
                head_mov_30s_list.append(head_mov_num)

        for au in audio[i]["chunks"]:
            for ls, id in zip(au["parameters"]["emotion model logits"], emot_mean):
                mean_mas.append(mean_mass(ls, id).tolist())
                au["parameters"]["emotion model logits"] = mean_mas
                au["parameters"]["emo_change_count"] = emo_change
                if au['number'] == 0:
                    au["parameters"]["blink_count"] = eye_predict[0]
                    au["parameters"]["smile_count"] = smile_predict[0]
                    au["parameters"]["head_movement_count"] = head_mov_30s_list[0]
                elif au['number'] == 1:
                    au["parameters"]["blink_count"] = eye_predict[1]
                    au["parameters"]["smile_count"] = smile_predict[1]
                    au["parameters"]["head_movement_count"] = head_mov_30s_list[1]
                elif au['number'] == 2:
                    au["parameters"]["blink_count"] = eye_predict[2]
                    au["parameters"]["smile_count"] = smile_predict[2]
                    au["parameters"]["head_movement_count"] = head_mov_30s_list[2]

