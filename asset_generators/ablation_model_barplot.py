import numpy as np
import os
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'

client = MlflowClient()

def get_result(id):
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    return np.round(100*client.get_run(id).data.metrics["val_macro_f1"], 1)

def plot(experiment_name):
    with plt.style.context(['science', "ieee", "high-vis"]):
        fig, ax = plt.subplots()

        y_pos = np.arange(5)
        x_pos = np.arange(0.5, 1, 0.1)

        ax.barh(y_pos, x_pos)

        ax.set(ylabel='Model')
        ax.set(xlabel='Validation Macro-F1 Score')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0,5])

        alength = 0.06
        aprops = {'arrowstyle': '->', "shrinkA": 0, "shrinkB": 0, "lw": 0.5}

        fig.savefig(f"figures/{experiment_name}.pdf")

plot("test")