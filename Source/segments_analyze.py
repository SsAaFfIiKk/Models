import os
import pickle
import json
# В этом модуля наши функции из комбайна, я просто убрал у них проверку на длинну
from agregation import *
# Модуль для загрузки моделей
from Source.Model_loader import *
# Модуль подготовит видос для обработки по чанкам
from Source.Video_preparation import *
# Модуль для анализа кадров
from Source.Video_analyze import *

# Определяем девайс и пути до моделей и заграем их
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_eye = "F:/Python/Models/pth/eyeB_91.8.pth"
path_to_smile = "F:/Python/Models/pth/smileB_90.8.pth"
path_to_emot = "F:/Python/Models/pth/res18_emot_67.4.pth"

loader = ModelLoader(device, path_to_eye, path_to_smile, path_to_emot)
loader.load_cv_model()
loader.load_torch_models()
model_eye, model_smile, model_emot, model_cv = loader.get_models()

# Пока что просто собираю все сегменты которые лежат в отдельной папки
path_to_videos = "F:/Python/Data/SegmentsA"
video_list = [os.path.join(path_to_videos, name) for name in os.listdir(path_to_videos) if name.endswith(".mp4")]
chunk_size = 30

# Загруженные модели передаем классу анализа для работы
analyzer = VideoAnalyze(model_eye, model_smile, model_emot, model_cv)
analyzer.prepare_transforms()

folder_param = "F:/Python/Models/Source/param"
folder_emot = "F:/Python/Models/Source/emot/video"

def analyze_frame(cor):
    centrs = analyzer.get_face_position(cor)
    eye_predict = analyzer.get_eye_predict()
    smile_predict = analyzer.get_smile_predict()
    emot_predict = analyzer.get_emot_predict()
    return centrs, eye_predict, smile_predict, emot_predict


def analyze_video(v_num, cap, chunks_borders):
    name_param = v_num + "_video_parameters.json"
    name_emot = v_num + "_video_emotion_logits.json"
    segment_video_parameters = {}
    segment_video_emot_logits = {}

    segment_video_parameters["segment"] = {}
    segment_video_emot_logits["segment"] = {}

    segment_video_parameters["segment"]["id"] = v_num
    segment_video_emot_logits["segment"]["id"] = v_num

    chunk_param_list = []
    chunk_emot_list = []

    chunk_count = 1
    # log = {}
    # начинаем перебор по чанкам
    for border in chunks_borders:
        chunk_pram_dict = {}
        chunk_emot_dict = {}

        chunk_pram_dict["number"] = chunk_count
        chunk_emot_dict["number"] = chunk_count

        predicts = []

        centrs = []
        eye_predicts = []
        smile_predicts = []
        emot_predicts = []

        frame_start = border[0]
        frame_end = border[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        # Анализируем каждый кадр в чанке
        # log["segment_" + str(chunk_count)] = chunk_count
        print("Start analyze {} chunk".format(chunk_count))
        chunk_len = frame_end - frame_start

        for frame_ind in range(frame_start, frame_end):
            if frame_ind == frame_start:
                chunk_pram_dict["duration"] = chunk_len / fps
                chunk_emot_dict["duration"] = chunk_len / fps

            ret, frame = cap.read()
            if frame is not None:
                cor = analyzer.get_face_borders(frame)
            # Упрт проверка чтоб трекер не отрыгнуло
                if cor[0] < 0:
                    cor[0] = 0

                analyzer.check_face(frame, cor)
                # Если есть лицо передаем его моделям
                if analyzer.face_detect:
                    centr, eye, smile, emot = analyze_frame(cor)
                    centrs.append(centr)
                    eye_predicts.append(eye)
                    smile_predicts.append(smile)
                    emot_predicts.append(emot)


                    predicts.append({'head_pos': centr,
                                     'emo_class': emot,
                                     'blink': eye,
                                     'smile': smile
                                     })

        # Запись логов в исходном виде
        # log["frames_for_{}_chunk".format(str(chunk_count))] = predicts
        # with open("video_" + str(v_num) + "_chunk_" + str(chunk_count) + ".data", "wb") as f:
        #     pickle.dump(log, f)

        print("Start aggregation")
        head_move, eye_mean, smile_mean, emot_mean = perform_aggregation(fps,
                                                                         chunk_len,
                                                                         centrs,
                                                                         eye_predicts,
                                                                         smile_predicts,
                                                                         emot_predicts)
        chunk_pram_dict["num_head_movenents"] = head_move[0]
        chunk_pram_dict["num_blinks"] = eye_mean[0]
        chunk_pram_dict["num_smiles"] = smile_mean[0]
        chunk_emot_dict["video_emotion_model_logits"] = emot_mean

        chunk_param_list.append(chunk_pram_dict)
        chunk_emot_list.append(chunk_emot_dict)

        chunk_count += 1

    segment_video_parameters["segment"]["chunks"] = chunk_param_list
    segment_video_emot_logits["segment"]["chunks"] = chunk_emot_list

    with open(os.path.join(folder_param, name_param), "w") as file:
        json.dump(segment_video_parameters, file, indent=2)

    with open(os.path.join(folder_emot, name_emot), "w") as file:
        json.dump(segment_video_emot_logits, file, indent=2)


# Перебераем все видоса из списка. Каждый делится на чанкии анализируется полностью
for video in video_list:
    num = video.split("\\")[1].split("_")[0].split(".")[0]
    print("Analyze video number {}".format(num))
    prep = VideoPreparation(video, chunk_size)
    prep.make_chunks_borders()
    cap, chunks_bordesrs, fps = prep.get_data()
    analyze_video(num, cap, chunks_bordesrs)