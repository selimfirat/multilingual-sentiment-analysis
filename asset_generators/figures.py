import torch
import numpy as np
import os
import pickle

import plotly.express as px
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage

from sklearn.metrics import multilabel_confusion_matrix

"""
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
"""

def corr_fig(corr, emojis_dict):
    fig = px.imshow(corr, x=emojis_dict, y=emojis_dict, color_continuous_scale='oryel')
    fig.update_xaxes(side="top")
    fig.show()

def cluster_fig(y_preds, emojis_dict):
    #fig = ff.create_dendrogram(np.transpose(y_preds), labels=emojis_dict)
    fig = ff.create_dendrogram(np.transpose(y_preds), labels=emojis_dict, orientation='bottom',
        linkagefun=lambda x: linkage(np.transpose(y_preds), 'average', metric='euclidean')
    )
    fig.update_layout(width=800, height=500)
    fig.show()
    
def conf_mat(y_preds, y_trues, emojis_dict):
    mcm = multilabel_confusion_matrix(y_trues, y_preds)
    print(mcm.shape)
    fig = ff.create_annotated_heatmap(mcm, x=emojis_dict, y=emojis_dict, colorscale='Viridis')
    fig.show()

num_emojis = 32

y_preds = (torch.load("data/y_test_pred.pt"))
corr = np.corrcoef(np.transpose(y_preds))

pkl_path = "data/emojis_zipf.pkl"
if os.path.exists(pkl_path):
    emojis_dict = pickle.load(open(pkl_path, "rb"))
    emojis_dict = list(emojis_dict.keys())[:num_emojis]
    print(emojis_dict)

    #corr_fig(corr, emojis_dict)
    cluster_fig(y_preds, emojis_dict)
    #conf_mat(y_preds, y_preds, emojis_dict)
    
else:
    print('emojis_zipf.pkl not found :(')
    
