import os
import numpy as np


def get_result(id):
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    return str(np.round(100*client.get_run(id).data.metrics["val_macro_f1"], 1))


table = """
\\begin{table}[htbp!]
\\caption{F1-Macro scores of different network architectures on the validation set.}
\\centering
\\begin{tabular}{@{}lc@{}}
\\toprule
""" + \
f"""
    Network & Validation F1-Macro \\\\ \\midrule
    Bidirectional LSTM & {get_result("bc13747099284e15a689a942c526a87f")} \\\\ \\midrule
    Bidirectional LSTM $\\backslash$w CE & {get_result("1de20edaa5e04ca99a3ac344865aca12")} \\\\
    Unidirectional LSTM & {get_result("0c14a0d63d714dab842ba877ee6a1971")} \\\\
    Bidirectional RNN & {get_result("3914954887144a6fabfbd406b0cf81f5")} \\\\
    Bidirectional GRU & {get_result("2e921672656b4c77a43931354ca5b917")} \\\\
""" + """XML-CNN~\\cite{liu2017deep} & 
    """ + get_result("0d33a104f14745ce83efa45da6740593") + """\\\\
     \\bottomrule
\\end{tabular}
\\label{tab:ablation_model}
\\end{table}
"""

target_path = "figures/tables"
if not os.path.exists(target_path):
    os.makedirs(target_path)

print(os.path.join(target_path, "ablation_model.tex"))
print(table)
f = open(os.path.join(target_path, "ablation_model.tex"), "w+", encoding="utf-8")
f.write(table)
f.close()