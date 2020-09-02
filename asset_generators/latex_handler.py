import torch
import pickle

class LatexHandler:

    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets = {
            'SE0714': ('data/SE0714/raw.pickle', 3, 'Emotion', 'Headlines'),
            'Olympic': ('data/Olympic/raw.pickle', 4, 'Emotion', 'Tweets'),
            'PsychExp': ('data/PsychExp/raw.pickle', 7, 'Emotion', 'Experiences'),
            'SS-Twitter': ('data/SS-Twitter/raw.pickle', 2, 'Sentiment', 'Tweets'),
            'SS-Youtube': ('data/SS-Youtube/raw.pickle', 2, 'Sentiment', 'Video Comments'),
            'SCv1': ('data/SCv1/raw.pickle', 2, 'Sarcasm', 'Debate Forums'),
            'SCv2-GEN': ('data/SCv2-GEN/raw.pickle', 2, 'Sarcasm', 'Debate Forums'),
            "kaggle-insults": ('data/kaggle-insults/raw.pickle', 2, 'Sentiment', 'Video Comments'),
            "emoji-tweets": ("data/emoji-tweets/raw.pickle", 64, '-', '-')
        }
        print(self.get_table_template())
        
    def get_table_template(self, caption="<CAPTION>", label="<LABEL>"):
        tab_start = """
        \\begin{table*}[t]
        \caption{""" + caption + """}
        \\centering
        \\begin{tabular}{llllllll}
        \\toprule
        Name & Task & Domain & \\# Classes & Multi-Label & \\# Train & \\# Val & \\# Test \\\\ \midrule
        """
        
        rows = ""
        for ds in self.cfg['all_datasets']:
            data_path, num_classes, task, domain = self.datasets[ds]
            data = pickle.load(open(data_path, "rb"), encoding="utf-8", fix_imports=True)
            row = ds + " & " + task + " & " + domain + " & " + str(num_classes) + " & "
            
            if ds in self.cfg['ml_datasets']:
                row = row + "Yes" + " & "
            else:
                row = row + "No" + " & "
            
            row = row + str(len(data['train_ind'])) + " & " + str(len(data['val_ind'])) + " & " + str(len(data['test_ind'])) + \
            " \\\\ "
            rows = rows + row + "\n"
        
        tab_end = """
        \\bottomrule
        \\end{tabular}
        \label{""" + label + """}
        \\end{table*}
        """
        return tab_start + rows + tab_end
    
    """
    def get_row(self, ds, data, col_names):
        ss = ds + " & "
        for cn in col_names:
            ss = ss + data[cn] + " & "
        ss = ss[: -2] + "  \\ \hline"
        return ss
        
    """
