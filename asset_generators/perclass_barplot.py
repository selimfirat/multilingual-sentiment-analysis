import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, accuracy_score, precision_score

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'


def plot(experiment_name):
    ft = "f7952febebc64a3282f6178b60417c81"
    y = torch.load(os.path.join("pretrained", ft, "y_val.pt"))*1
    y_pred = torch.load(os.path.join("pretrained", ft, "y_val_pred_labels.pt"))*1
    classes = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness",
               "surprise", "trust"]

    scores = []
    for i in  range(11):
        f1 = f1_score(y[:, i], y_pred[:, i])
        f1 = np.round(f1*100, 1)
        print(classes[i], f1)
        scores.append(f1)

    with plt.style.context(['science', "ieee", "high-vis"]):
        fig, ax = plt.subplots()
        colors = ["#FF0000", "#FFA854", "#FF54FF", "#009600", "#FFFF54", "#AAFF54", "#FFD454", "#5587FF", "#5151FF", "#59BDFF", "#54FF54"]
        y_pos = np.arange(len(classes))
        ax.bar(y_pos, scores, color=colors)
        ax.set_ylim([0.0, 100])
        plt.xticks(y_pos, classes, rotation=80)
        ax.set(ylabel='Validation F1 Score')
        ax.set(xlabel='Class')

        fig.savefig(f"figures/{experiment_name}.pdf")


plot("perclass_barplot")
