import os
import librosa
import winsound
import soundfile as sf


class Datamaker:
    def __init__(self, path_to_data, path_to_save):
        self.count_a = 0
        self.count_s = 0
        self.count_n = 0
        self.count_h = 0
        self.path = path_to_data
        self.path_to_save = path_to_save

    def load_audio(self, audio):
        return winsound.PlaySound(audio, winsound.SND_FILENAME)

    def save_audio(self, audio):
        au, _ = librosa.load(os.path.join(self.path, audio), 48000)
        label = ""
        print("Какая эмоция?")
        print("1-Angry")
        print("2-Sad")
        print("3-Neutral")
        print("4-Happy")

        while label == "":
            nm = int(input())
            if nm == 1:
                label = "Angry"
                file_name = os.path.join(self.path_to_save, str(self.count_n) + label + ".wav")
                self.count_n += 1
                sf.write(os.path.join(self.path_to_save, file_name), au, 48000, "PCM_24")
            elif nm == 2:
                label = "Sad"
                file_name = os.path.join(self.path_to_save, str(self.count_s) + label + ".wav")
                self.count_s += 1
                sf.write(os.path.join(self.path_to_save, file_name), au, 48000, "PCM_24")
            elif nm == 3:
                label = "Neutral"
                file_name = os.path.join(self.path_to_save, str(self.count_n) + label + ".wav")
                self.count_n += 1
                sf.write(os.path.join(self.path_to_save, file_name), au, 48000, "PCM_24")
            elif nm == 4:
                label = "Happy"
                file_name = os.path.join(self.path_to_save, str(self.count_h) + label + ".wav")
                self.count_h += 1
                sf.write(os.path.join(self.path_to_save, file_name), au, 48000, "PCM_24")
            else:
                print("Повторите ввод: ")


path = "F:/Python/Data/Audio/seg"
path_to_save = "F:/Python/Data/Audio"
chunks = os.listdir(path)
datamaker = Datamaker(path, path_to_save)
play = False

for chunk in chunks:
    play = True
    while play:
        print("1-Воспроизвести аудио")
        print("2-Сохранить")
        print("3-Следующий фрагмент")
        select = int(input("Chose one: "))
        if select == 1:
            datamaker.load_audio(os.path.join(path, chunk))
        elif select == 2:
            datamaker.save_audio(chunk)
            play = False
        elif select == 3:
            play = False
