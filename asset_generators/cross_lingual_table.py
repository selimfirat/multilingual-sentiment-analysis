import numpy as np
from mlflow.tracking import MlflowClient
client = MlflowClient()

def get_result(id, metric):

    return str(np.round(100*client.get_run(id).data.metrics[metric], 1))

metric = "val_macro_f1"

methods = {
    "Arabic (SP)": {
        "Arabic": {
            "mlflow": "92e052af4aa24f94bf07319043cbf6cc"
        },
        "English": {
            "mlflow": "eb96f4e56c18498d9851fd81b148eec1"
        },
        "Spanish": {
            "mlflow": "4042b97258dc4eeebb542256ecc8edd8"
        },
    },
    "English (EN)": {
        "Arabic": {
            "mlflow": "e55c80391bc842888f7069d6da60b294"
        },
        "English": {
            "mlflow": "40f9b2c34e1e4d29a4125b7a45d068fb"
        },
        "Spanish": {
            "mlflow": "b9f4041fca9d40e88978199b8628aebd"
        },
    },
    "Spanish (SP)": {
        "Arabic": {
            "mlflow": "c7c4441d50bd482b9032862670e5dbc1"
        },
        "English": {
            "mlflow": "17a7b70cb51b409f82cb990298ac6eb0"
        },
        "Spanish": {
            "mlflow": "682f622008b64d8c9d4ab382329c486d"
        },
    },
    "AR + EN": {
        "Arabic": {
            "mlflow": "ac777b065b8044cfaed88ee6458f1727"
        },
        "English": {
            "mlflow": "94883cc428b94352a36da705c3959b10"
        },
        "Spanish": {
            "mlflow": "c6ee856b47774ec3ae6585261a4eb2b8"
        },
    },
    "AR + SP": {
        "Arabic": {
            "mlflow": "a121e91227304d4b89e24f4ffd36e1a0"
        },
        "English": {
            "mlflow": "adeef08a76ae4a7a9a51a057b97d309e"
        },
        "Spanish": {
            "mlflow": "10546853a48d42a3890d273e57b09fcd"
        },
    },
    "EN + SP": {
        "Arabic": {
            "mlflow": "3d31fc846a7847b6ba846b8fc579c129"
        },
        "English": {
            "mlflow": "ea5c8d73f93f4cfe8aa57c8467e785e3"
        },
        "Spanish": {
            "mlflow": "dbf9580c64e14ff2bbbef141ea98f59d"
        },
    },
    "EN + AR + SP": {
        "Arabic": {
            "mlflow": "9d07da75ccb04999b3397d1c4222166a"
        },
        "English": {
            "mlflow": "28ee454004574bf7884de065c813e3a7"
        },
        "Spanish": {
            "mlflow": "f91a9800f766487389bfd51d6e12305c"
        },
    }
}

res = ""
for name, method in methods.items():
    res += f"{name} & "
    scores = []
    for data in ["Arabic", "English", "Spanish"]:
        if data in method:
            if "mlflow" in method[data] and method[data]["mlflow"] != "":
                scores.append(get_result(method[data]["mlflow"], metric))
            else:
                scores.append("-")

    res += " & ".join(scores)

    res += " \\\\"
    if name == "Ours":
        res += "\\midrule"
    res += "\n"

print(res)