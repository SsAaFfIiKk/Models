import pickle
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
            cor = analyzer.get_face_borders(frame)
            if cor[0] < 0:
                cor[0] = 0

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
        with open("chunk_" + str(chunk_count) + ".data", "wb") as f:
            pickle.dump(log, f)
        chunk_count += 1


analyze_video(cap, chunks_bordesrs)
