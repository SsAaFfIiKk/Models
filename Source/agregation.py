import numpy as np


def perform_aggregation(fps, heads, blinks, smiles,  emotions):
    head_arr = np.array(heads).astype(np.float64)
    emot_mean = []
    head_mov_30s_list = []
    step = 30 * fps

    eye_predict = compute_states_on_30s(blinks, step)
    smile_predict = compute_states_on_30s(smiles, step)

    for idx_30s_chunk in range(0, len(head_arr), step):
        head_mov_num = get_head_movs_num(head_arr[idx_30s_chunk:idx_30s_chunk + step], fps)
        head_mov_30s_list.append(head_mov_num)

    for count in emotions:
        three_s = get_chunks(count, 3 * fps)
        for pr in three_s:
            emot_mean.append(np.mean(pr, axis=0).reshape(-1).tolist())

    return head_mov_30s_list, eye_predict, smile_predict, emot_mean


def get_chunks(lst, chunk_size):
    """
    разделяем список на 30 секунд,
    """
    list_30s = []
    for idx_30s in range(0, len(lst), chunk_size):
        list_30s.append(lst[idx_30s:idx_30s+chunk_size])
    return list_30s

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