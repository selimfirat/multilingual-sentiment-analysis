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
    Network & F1-Macro \\\\ \\midrule
    \\midrule
    Cross Entropy (CE) & {get_result("ac0c97cd88fd421b9dba4b66c47c987e")} \\\\""" + """
    Focal~\cite{lin2017focal} &""" + get_result("7b74da6695634ddfadd5351ea580f243") + """\\\\ 
""" + """
    XML-CNN~\\cite{liu2017deep} & 
    """ + get_result("c70b8a6d1d754bce94467f79c921ac24") + """\\\\
     \\bottomrule
\\end{tabular}
\\label{tab:ablation_model}
\\end{table}
"""

target_path = "figures/tables"
if not os.path.exists(target_path):
    os.makedirs(target_path)

print(os.path.join(target_path, "ablation_model.tex"))
f = open(os.path.join(target_path, "ablation_model.tex"), "w+", encoding="utf-8")
f.write(table)
f.close()