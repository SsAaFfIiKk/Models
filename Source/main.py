import os
import pickle
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
path_to_videos = "F:/Python/Data/Segments"
video_list = [os.path.join(path_to_videos, name) for name in os.listdir(path_to_videos) if name.endswith(".mp4")]
chunk_size = 30

# Загруженные модели передаем классу анализа для работы
analyzer = VideoAnalyze(model_eye, model_smile, model_emot, model_cv)
analyzer.prepare_transforms()


def analyze_frame(cor):
    centrs = analyzer.get_face_position(cor)
    eye = analyzer.get_eye_predict()
    smile = analyzer.get_smile_predict()
    emot_label, emot = analyzer.get_emot_predict()
    return centrs, eye, smile, emot_label, emot


def analyze_video(v_num, cap, chunks_borders):
    chunk_count = 0
    # начинаем перебор по чанкам
    for border in chunks_borders:
        log = {}
        predicts = []
        log["segment_" + str(chunk_count)] = chunk_count

        centrs = []
        eye_predicts = []
        smile_predicts = []
        emot_predicts = []
        emot_label_predicts = []

        frame_start = border[0]
        frame_end = border[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        # Анализируем каждый кадр в чанке
        for frame_ind in range(frame_start, frame_end):
            ret, frame = cap.read()
            cor = analyzer.get_face_borders(frame)
            # Упрт проверка чтоб трекер не отрыгнуло
            if cor[0] < 0:
                cor[0] = 0

            analyzer.check_face(frame, cor)
            # Если есть лицо передаем его моделям
            if analyzer.face_detect:
                centr, eye, smile, emot_label, emot = analyze_frame(cor)
                centrs.append(centr)
                eye_predicts.append(eye)
                smile_predicts.append(smile)
                emot_predicts.append(emot)
                emot_label_predicts.append(emot_label)

                predicts.append({'head_pos': centr,
                                 'emo_class': emot,
                                 'blink': eye,
                                 'smile': smile
                                 })
        # Запись логов в исходном виде
        log["frames_for_{}_chunk".format(str(chunk_count))] = predicts
        with open( "video_" + str(v_num) + "chunk_" + str(chunk_count) + ".data", "wb") as f:
            pickle.dump(log, f)

        print("Start aggregation")
        head_move, eye_mean, smile_mean, emot_mean = perform_aggregation(fps,
                                                                         centrs,
                                                                         eye_predicts,
                                                                         smile_predicts,
                                                                         emot_predicts)

        chunk_count += 1

# Перебераем все видоса из списка. Каждый делится на чанкии анализируется полностью
for num, video in enumerate(video_list):
    print("Analyze video number {}".format(num))
    prep = VideoPreparation(video, chunk_size)
    prep.make_chunks_borders()
    cap, chunks_bordesrs, fps = prep.get_data()
    analyze_video(num, cap, chunks_bordesrs)
