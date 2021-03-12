import pandas as pd
import  matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def read_csv(path_to_file):
    df = pd.read_csv(path_to_file)
    return df


def convert_to_ind(emot_list, index_dict):
    ind_list = []
    for emot in emot_list:
        if emot in index_dict:
            ind_list.append(index_dict[emot])
    return ind_list


emotion_ind = {
     "Angry": -1,
     "Happy": 1,
     "Neutral": 0,
     "Sad": 0,
     "Scared": -1
}

path_to_csv = "F:/Python/Data/Audio/audio_only_ivan.csv"

if __name__ == "__main__":
    emotions = read_csv(path_to_csv)
    x = emotions["start_time"].values.tolist()
    y = emotions["audio_emotion_prediction"].values.tolist()
    # fig = go.FigureWidget([go.Scatter(x=x, y=y, mode="markers")])
    fig = px.scatter(emotions, x=x, y=y, color="prediction_confidence")
    fig.show()

    # ind_list = convert_to_ind(emotions["audio_emotion_prediction"].values.tolist(), emotion_ind)
    # position = [-1, 0, 1]
    #
    # pos_x = []
    # pos_y = []
    #
    # neg_x = []
    # neg_y = []
    #
    # zeros_x = []
    # zeros_y = []
    #
    # for x, y in zip(range(len(ind_list)), ind_list):
    #     if y > 0:
    #         pos_x.append(x)
    #         pos_y.append(y)
    #     elif y < 0:
    #         neg_x.append(x)
    #         neg_y.append(y)
    #     else:
    #         zeros_x.append(x)
    #         zeros_y.append(y)
    #
    # fig, graph_axes = plt.subplots()
    #
    # graph_axes.scatter(pos_x, pos_y, c='g')
    # graph_axes.scatter(neg_x, neg_y, c='r')
    # graph_axes.scatter(zeros_x, zeros_y, c='b')
    #
    # graph_axes.set_yticks(position)
    # graph_axes.set_yticklabels(["Negative", "Neutral", "Positive"])
    # plt.show()
