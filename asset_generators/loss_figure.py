import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'


models = {
    "465acf71d51c41ff904e867b252e6e03": { #f46db724d2fa480b9f271c810be59372 (with 0.05 weight smoothing)
        "name": "Loss Equalization"
    },
    "b27fdeb73f8e4369a04ea92b50274e10": {
        "name": "Uniform"
    },
    "7f41f745e53d4a9596f382d66eb5465c": {
        "name": "Inverse Frequency"
    },
    "f9d2533d21304721b9f6c3b8fb416821": {
        "name": "Class Balanced"
    },
}
"""
"6232094a72474fab821eb00e2f74ddd3": {
    "name": "Cost Sensitive"
}
"""

long_epoch_indices = []
min_num_batchs = float("inf")
for model_idx, model in models.items():
    class_losses_tuple = torch.load(f"pretrained/{model_idx}/class_losses.pt")
    losses = []
    total_items = 0
    last_epoch = 0
    epoch_indices = []
    num_batchs = 0
    for idx, (epoch_idx, batch_size, class_loss) in enumerate(class_losses_tuple):
        losses.append(class_loss)
        total_items += batch_size
        num_batchs += 1
        if last_epoch != epoch_idx:
            last_epoch += 1
            epoch_indices.append(idx)

    if len(epoch_indices) > len(long_epoch_indices):
        long_epoch_indices = epoch_indices
    min_num_batchs = min(num_batchs, min_num_batchs)

    losses = np.array(losses)
    mean_loss = np.expand_dims(losses.mean(axis=1), axis=1)
    mean_loss = np.repeat(mean_loss, repeats=losses.shape[1], axis=1)
    losses = ((losses - mean_loss)**2).sum(axis=1)
    model["losses"] = losses

num_train_samples = min_num_batchs / len(long_epoch_indices)

for model_idx, model in models.items():
    model["losses"] = np.cumsum(model["losses"])

with plt.style.context(['science', "ieee", "high-vis"]):
    fig, ax = plt.subplots()
    for model_idx, model in models.items():
        labels = np.zeros(model["losses"].shape[0])
        ax.plot(np.arange(0, model["losses"].shape[0], dtype=np.float)/num_train_samples, model["losses"], label=model["name"])

    ax.legend(title='', loc="lower right")
    #plt.xticks(np.arange(0, len(long_epoch_indices)+1, 5, dtype=np.float))
    ax.set(ylabel='Cumulative MSE')
    ax.set(xlabel='Epochs')

    fig.savefig("figures/loss_figure.pdf")
