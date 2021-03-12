import wave
import json
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)

path_to_model = "F:/Python/Data/model_data/model"

wf = wave.open("F:/Python/Data/Audio/segments/6.wav", "rb")
model = Model(path_to_model)
rec = KaldiRecognizer(model, wf.getframerate())

result = ""
last_n = False

while True:
    data = wf.readframes(wf.getframerate())
    if len(data) == 0:
        break

    # if rec.AcceptWaveform(data):
    #     print(rec.Result())
    #
    # else:
    #     print(rec.PartialResult())

    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())

        if res['text'] != '':
            result += f" {res['text']}"
            last_n = False
        elif not last_n:
            result += '\n'
            last_n = True

res = json.loads(rec.FinalResult())
result += f" {res['text']}"

print(result)
