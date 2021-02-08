import numpy as np
from Source.Model_loader import *
from Source.Video_preparation import *
from Source.Video_analyze import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_eye = "F:/Python/Models/pth/eyeB_91.8.pth"
path_to_smile = "F:/Python/Models/pth/smileB_90.8.pth"
path_to_emot = "F:/Python/Models/pth/res18_emot_67.4.pth"

loader = ModelLoader(device, path_to_eye, path_to_smile, path_to_emot)
loader.load_cv_model()
loader.load_torch_models()
model_eye, model_smile, model_emot, model_cv = loader.get_models()

path_to_video = "F:/Python/Data/demo/video.mp4"
chunk_size = 30
prep = VideoPreparation(path_to_video, chunk_size)
prep.make_chunks_borders()
cap, chunks_bordesrs = prep.get_data()

analyzer = VideoAnalyze(model_eye, model_smile, model_emot, model_cv)
analyzer.prepare_transforms()


def analyze_frame(cor):
    centrs = analyzer.get_face_position(cor)
    eye = analyzer.get_eye_predict()
    smile = analyzer.get_smile_predict()
    emot = analyzer.get_emot_predict()
    return centrs, eye, smile, emot


# def get_face_borders(frame):
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     model_cv.setInput(blob)
#     detections = model_cv.forward()
#
#     for found in range(0, detections.shape[2]):
#         box = detections[0, 0, found, 3:7] * np.array([w, h, w, h])
#         cor = box.astype("int")
#         cor = cor
#
#         return cor


def analyze_video(cap, chunks_borders):
    chunk_count = 0
    for border in chunks_borders:
        log = {}
        predicts = []
        log["segment_" + str(chunk_count)] = chunk_count

        centrs = []
        eye_predicts = []
        smile_predicts = []
        emot_predicts = []

        frame_start = border[0]
        frame_end = border[1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        for frame_ind in range(frame_start, frame_end):

            ret, frame = cap.read()
            cor = get_face_borders(frame)
            analyzer.check_face(frame, cor)

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
        log["frames"] = predicts
        chunk_count += 1


analyze_video(cap, chunks_bordesrs)
