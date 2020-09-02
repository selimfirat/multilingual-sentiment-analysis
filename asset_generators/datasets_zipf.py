import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from adjustText import adjust_text

from dataset import TextClassificationDataset

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'

datasets = {
    "SemEval_English": {
        "name": "English",
    },
    "SemEval_Arabic": {
        "name": "Arabic",
    },
    "SemEval_Spanish": {
        "name": "Spanish"
    }
}

classes = np.array(["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise","trust"])

for data_idx, data in datasets.items():
    cfg = {
        "data": data_idx,
        "pretrained": "xlm-roberta-large",
    }
    train = TextClassificationDataset(cfg, "train")
    samples_per_cls = train.samples_per_cls
    sorted_idx = np.argsort(-samples_per_cls)
    data["samples_per_cls"] = samples_per_cls[sorted_idx]
    data["classes"] = classes[sorted_idx]
    data["num_classes"] = train.num_classes
    print(data["classes"])

with plt.style.context(['science', "ieee", "high-vis", "scatter", "no-latex"]):
    fig, ax = plt.subplots()
    texts = []
    for data_idx, data in datasets.items():
        ax.plot(range(1, data["num_classes"]+1), data["samples_per_cls"], label=data["name"])
        for i in range(data["num_classes"]):
            texts.append(ax.text(i+1, data["samples_per_cls"][i], data["classes"][i].title(), ha="center", va="center", fontsize=4))

    adjust_text(texts, autoalign='y',
            only_move={'points':'y', 'text':'y'}, force_points=0.5)
    ax.legend(title='', loc="upper right")
    #plt.xticks(np.arange(0, len(long_epoch_indices)+1, 5, dtype=np.float))
    ax.set(ylabel='Label Frequency')
    ax.set(xlabel='Label Rank')

    fig.savefig("figures/datasets_zipf.pdf")
