import numpy as np
from scipy import stats


def get_key(dct):
    # Получаем ключи из видео.datа
    keys = []
    for key in dct.keys():
        keys.append(key)
    return keys


def get_chunks(lst, chunk_size):
    """
    разделяем список на 30 секунд,
    если длинна меньше 30 то пропускаем его
    """
    list_30s = []
    for idx_30s in range(0, len(lst), chunk_size):
        if len(lst[idx_30s:idx_30s+chunk_size]) == chunk_size:
            list_30s.append(lst[idx_30s:idx_30s+chunk_size])
    return list_30s


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

        # !!!!!!!!!!!!
        # Критерий выбран пока произвольно!
        head_mov_num += (dists > 20).sum()
        # compute the confidence interval according to Sudents t-test
        #conf = stats.t.interval(0.995, df=len(dists)-1, loc=dists.mean(), scale=stats.sem(dists))
        # filter distances that don't lay in the confidence iterval
        #head_mov_num += (dists < conf[0]).sum() + (dists > conf[1]).sum()

    return int(head_mov_num)


def get_emo_class(mean_emo):
    EMOTIONS_LIST = ["Angry", "Happy", "Neutral", "Sad", "Scared"]
    class_lst = []
    for i in mean_emo:
        class_lst.append(EMOTIONS_LIST[np.argmax(i)])

    return class_lst


def mean_mass(fst, scd):
    '''
    Функция для усреденения лождитов из аудио и видео
    Входные параметры это два списка списков предиктов
    Затем она вызывает функцию получения класса эмоции
    и возвращает список из 10 эмоций
    '''
    mean_mas = []
    for ls_a, ls_v in zip(fst, scd):
        mean_mas.append((np.array(ls_a) + np.array(ls_v)) / 2.0)
    if len(mean_mas) == 10:
        return  get_emo_class(mean_mas)


def base_change_count(lst):
    # Функуия получает списко из 10 эиоций и считает кол-во измений  эмоций
    base_conut = 0
    for pas, pres in zip(lst, lst[1:]):
        if pas != pres:
            base_conut += 1

    return base_conut


def neutral_change_count(lst):
    # функция для подсчёта переходов через нетраль
    neutral = "Neutral"
    emo_count = 0
    for i in range(1, len(lst) - 1):
        if neutral != lst[i - 1] and lst[i] == neutral and lst[i + 1] != neutral:
            emo_count += 1

    return  emo_count


def compose_results(results_audio, results_video):
        # узнаем каие сегменты будем обрабатывать
        intervals = get_key(results_video)

        # Задаем шаг смещения и фпс
        # Они нужны для разбиения предиктов на чанки и их усреднения
        fps = int(results_video[intervals[-1]])
        step = 30 * fps

        for i in range(len(intervals) - 1):
            emot_mean = []
            head_mov_30s_list = []

            # получаеть все предикты для сегментв
            emot = results_video[intervals[i]]["emot"]
            # делим на чанки
            thirty_s = get_chunks(emot, 30 * fps)
            # усредняем их
            for count in thirty_s:
                three_s = get_chunks(count, 3 * fps)
                for pr in three_s:
                    emot_mean.append(np.mean(pr, axis=0).reshape(-1).tolist())

            # получаем предикты для глаз
            eye = results_video[intervals[i]]["eye"]
            # конвертируем в метки
            eye_labels_list = get_labels(eye)
            # получаем количество на каждом 30-секундном интервале
            eye_predict = compute_states_on_30s(eye_labels_list, step)

            # получаем предикты для улыбок
            smile = results_video[intervals[i]]["smile"]
            # конвертируем в метки
            smile_labels_list = get_labels(smile)
            # получаем количество на каждом 30-секундном интервале
            smile_predict = compute_states_on_30s(smile_labels_list, step)

            head_arr = np.array(results_video[intervals[i]]['head_position']).astype(np.float64)
            # получаем количество на 30-секундном интервале

            for idx_30s_chunk in range(0, len(head_arr), step):
                # если отрезок меньше 30 с, не считаем его
                if len(head_arr[idx_30s_chunk:idx_30s_chunk + step]) == step:
                    head_mov_num = get_head_movs_num(
                        head_arr[idx_30s_chunk:idx_30s_chunk + step], fps)
                    head_mov_30s_list.append(head_mov_num)

            # читаем каждый вложенный словарь
            for au in results_audio[i]["chunks"]:
                # усреднение предиктов аудио и видио и преобразование их в класы
                emot_class = mean_mass(au["parameters"]["emotion model logits"], emot_mean)
                # подсчет изменения относительно предыдущего состояния
                base_change = base_change_count(emot_class)
                # подсчет перехода через нетраль
                neutral_change = neutral_change_count(emot_class)

                au["parameters"]["predicted_emotions"] = emot_class
                au["parameters"]["emo_change_count"] = base_change
                au["parameters"]["emo_cross_neutral"] = neutral_change

                numb = au['number']
                au["parameters"]["blink_count"] = eye_predict[numb]
                au["parameters"]["smile_count"] = smile_predict[numb]
                au["parameters"]["head_movement_count"] = head_mov_30s_list[numb]

                # удаление ненужных данных
                del au["parameters"]["emotion model logits"]


        return results_audio

