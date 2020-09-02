import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mlflow.tracking import MlflowClient

from adjustText import adjust_text

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'


def get_result(id):
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    return np.round(100*client.get_run(id).data.metrics["val_macro_f1"], 1)


client = MlflowClient()


def get_points(experiment_name, weights, param):
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs([experiment.experiment_id])
    scores = []
    params = []
    for run in runs:
        data = run.data
        if "val_macro_f1" not in data.metrics:
            continue
        if data.params["weights"] == weights:
            params.append(float(data.params[param]))
            scores.append(data.metrics["val_macro_f1"])

    indices = np.argsort(params)
    params = np.array(params)[indices]
    scores = np.array(scores)[indices]

    scores = np.round(scores*100, 1)

    return params, scores


def get_score(experiment_name, weights):
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs([experiment.experiment_id])
    largest_score = -1
    for run in runs:
        data = run.data
        if "val_macro_f1" not in data.metrics:
            continue
        if data.params["weights"] == weights:
            largest_score = max(largest_score, np.round(data.metrics["val_macro_f1"]*100, 1))

    return largest_score


def plot(experiment_name):
    dynamic_focal_params, dynamic_focal_scores = get_points(experiment_name, "dynamic_loss_size", "weight_smoothing")
    uniform_score = get_score(experiment_name, "uniform")
    inverse_score = get_score(experiment_name, "inverse")
    cost_sensitive_score = get_score(experiment_name, "cost_sensitive")
    class_balanced_params, class_balanced_scores = get_points(experiment_name, "class_balanced", "beta")

    print(dynamic_focal_scores, dynamic_focal_params)
    print(class_balanced_params, class_balanced_scores)
    with plt.style.context(['science', "ieee", "high-vis"]):
        fig, ax = plt.subplots()

        ax.plot(dynamic_focal_params, dynamic_focal_scores, label="Dynamic")
        ax.plot([0.0, 1.0], [uniform_score, uniform_score], label="Uniform")
        ax.plot([0.0, 1.0], [inverse_score, inverse_score], label="Inverse")
        ax.plot([0.0, 1.0], [cost_sensitive_score, cost_sensitive_score], label="Cost-Sensitive")
        ax.plot(class_balanced_params, class_balanced_scores, label="Class-Balanced")

        ax.set(ylabel='Validation Macro-F1 Score')
        ax.set(xlabel='Smoothing Coefficient $\kappa$')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([55, 65])

        alength = 0.06
        aprops = {'arrowstyle': '->', "shrinkA": 0, "shrinkB": 0, "lw": 0.5}

        ax.legend(title='')

        fig.savefig(f"figures/{experiment_name}.pdf")

plot("ablation_loss_plot")
plot("ablation_loss_plot2")
