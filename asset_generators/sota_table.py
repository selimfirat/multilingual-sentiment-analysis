import numpy as np
from mlflow.tracking import MlflowClient
client = MlflowClient()

def get_result(id, metric):

    return str(np.round(100*client.get_run(id).data.metrics[metric], 1))

methods = {
    "Ours": {
        "English": {
            "mlflow": "28ee454004574bf7884de065c813e3a7"
        },
        "Spanish": {
            "mlflow": "f91a9800f766487389bfd51d6e12305c"
        },
        "Arabic": {
            "mlflow": "9d07da75ccb04999b3397d1c4222166a"
        }
    },
    "Ours-SL": {
        "English": {
            "mlflow": "9b8dbce5a76647cb8d4eb83a30fdac44"
        },
        "Spanish": {
            "mlflow": "a54141fa0d5e41c280e2a1bfdf8699f3"
        },
        "Arabic": {
            "mlflow": "28b66120527f4530826c9b2b78c8ded3"
        }
    },
    "Tw-StAR~\\cite{mulki2018tw}": {
        "English": {
            "macro_f1": "45.2",
            "jaccard_index": "48.1",
            "micro_f1": "60.7"
        },
        "Spanish": {
            "macro_f1": "39.2",
            "jaccard_index": "43.8",
            "micro_f1": "52.0"
        },
        "Arabic": {
            "macro_f1": "44.6",
            "jaccard_index": "46.5",
            "micro_f1": "59.7"
        }
    },
    # Arabic
    "CA-GRU~\\cite{samy2018context}": {
        "Arabic": {
            "macro_f1": "49.5",
            "jaccard_index": "53.2",
            "micro_f1": "64.8"
        }
    },
    "HEF-DF~\\cite{alswaidan2020hybrid}": {
        "Arabic": {
            "macro_f1": "50.2",
            "jaccard_index": "51.2",
            "micro_f1": "63.1"
        }
    },
    "EMA~\\cite{badaro2018ema}": {
        "Arabic": {
            "macro_f1": "46.1",
            "jaccard_index": "48.9",
            "micro_f1": "61.8"
        }
    },
    "PARTNA": {
        "Arabic": {
            "macro_f1": "47.5",
            "jaccard_index": "48.4",
            "micro_f1": "60.8"
        }
    },
    # English
    "NTUA-SLP~\\cite{baziotis2018ntua}": {
        "English": {
            "macro_f1": "52.8",
            "jaccard_index": "58.8",
            "micro_f1": "70.1"
        }
    },
    "psyML~\\cite{gee2018psyml}": {
        "English": {
            "macro_f1": "57.4",
            "jaccard_index": "57.4",
            "micro_f1": "69.7"
        }
    },
    "NVIDIA~\cite{kant2018practical}": {
        "English": {
            "macro_f1": "56.1",
            "jaccard_index": "57.7",
            "micro_f1": "69.0"
        }
    },
    # Spanish
    "ELiRF-UPV~\\cite{gonzalez2019elirf}": {
        "Spanish": {
            "jaccard_index": "45.8",
            "micro_f1": "53.5",
            "macro_f1": "44.0"
        }
    },
    "MILAB\\_SNU": {
        "Spanish": {
            "jaccard_index": "46.9",
            "micro_f1": "55.8",
            "macro_f1": "40.7"
        }
    },
    "FastText~\\cite{fasttext}": {
        "English": {
            "macro_f1": "35.0",
            "jaccard_index": "25.5",
            "micro_f1": "39.9"
        },
        "Spanish": {
            "macro_f1": "27.0",
            "jaccard_index": "20.6",
            "micro_f1": "31.9"
        },
        "Arabic": {
            "macro_f1": "35.3",
            "jaccard_index": "25.5",
            "micro_f1": "40.2"
        }
    },
    "DeepMoji~\\cite{deepmoji}": {
        "English": {
            "macro_f1": "",
            "jaccard_index": "",
            "micro_f1": ""
        },
    },
}

res = ""
for name, method in methods.items():
    res += f"{name} & "
    scores = []
    for data in ["Arabic", "English", "Spanish"]:
        if data in method:
            if "mlflow" in method[data]:
                scores.append(get_result(method[data]["mlflow"], "macro_f1"))
                scores.append(get_result(method[data]["mlflow"], "micro_f1"))
                scores.append(get_result(method[data]["mlflow"], "jaccard_index"))
            else:
                scores.extend([method[data]["macro_f1"], method[data]["micro_f1"], method[data]["jaccard_index"]])
        else:
            scores.extend(["-", "-", "-"])
    res += " & ".join(scores)

    res += " \\\\"
    if name == "Ours":
        res += "\\midrule"
    res += "\n"

print(res)
