import os
import argparse
import torch
import numpy as np

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description='Sentiment Deep Metric Learning')

    parser.add_argument("--module", default="ml", type=str)
    parser.add_argument("--data", type=str, default="SS-Youtube")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--experiment", type=str, default="default")

    parser.add_argument("--model", default="lstm", type=str)
    parser.add_argument("--pretrained", type=str, default="xlm-roberta-large")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_center", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rec_dropout", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--pooling", type=str, default="attention")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--no_bidirectional", action="store_true", default=False)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--no_train", action="store_true", default=False)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--finetune_type", type=str, default="change_final")
    parser.add_argument("--freeze_bert", action="store_true", default=False)

    parser.add_argument("--loss", type=str, default="focal")

    parser.add_argument("--weights", type=str, default="dynamic_loss_size")
    parser.add_argument("--thresholding", type=str, default="class_specific")

    parser.add_argument("--num_bins", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda", type=float, default=0.003)
    parser.add_argument("--weight_smoothing", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.9999)
    parser.add_argument("--gamma", type=float, default=2.0)
    
    parser.add_argument("--prepro", type=str, default="all")

    cfg = vars(parser.parse_args())

    cfg["bidirectional"] = not cfg["no_bidirectional"]

    data = ['SCv2-GEN', 'PsychExp']
    cfg['all_datasets'] = ['SS-Twitter', 'SS-Youtube', 'SCv1', 'SCv2-GEN', 'SE0714', 'Olympic', 'SemEval_Arabic', 'SemEval_English', 'SemEval_Spanish', 'SemEval_Arabic_English', 'SemEval_Arabic_Spanish', 'SemEval_English_Spanish', 'SemEval_English_Turkish', 'SemEval_Arabic_English_Spanish', 'SemEval_Turkish', 'SemEval_Tran_Spanish']
    cfg['ml_datasets'] = ['SE0714', 'Olympic', 'emoji-tweets']

    if cfg["data"] != "all":
        data = [cfg["data"]]

    if not torch.cuda.is_available():
        cfg["device"] = "cpu"

    for datum in data:
        np.random.seed(61)
        torch.manual_seed(61)
        cfg["data"] = datum

        if cfg["module"] == "feature_extraction":
            from feature_extraction import FeatureExtraction

            fe = FeatureExtraction(cfg)
            fe.extract()
        elif cfg["module"] == "ml":
            print(f"Experimenting with the data {datum}")
            from experiment_ml import ExperimentMultiLabelClassification
            experiment = ExperimentMultiLabelClassification(cfg)
            experiment.run()
        elif cfg["module"] == "etp":
            from utils.emoji_tweets_preprocessing import EmojiTweetsPreprocessing
            etp = EmojiTweetsPreprocessing()
            etp.run()
            break
        elif cfg["module"] == "dataset_table":
            from asset_generators.latex_handler import LatexHandler
            LatexHandler(cfg)
        elif cfg["module"] == "bert":
            from experiment_bert import ExperimentBERT
            experiment = ExperimentBERT(cfg)
            experiment.run()
        elif cfg["module"] == "random":
            from experiment_random import ExperimentRandom
            experiment = ExperimentRandom(cfg)
            experiment.run()
        elif cfg["module"] == "emojis_zipf":
            from asset_generators.zipf_emojis import EmojisZipfLaw
            emojis_zipf = EmojisZipfLaw()
            emojis_zipf.run()
        elif cfg["module"] == "fasttext":
            from experiment_fasttext import ExperimentFasttext
            experiment = ExperimentFasttext(cfg)
            experiment.run()
        elif cfg["module"] == "fasttext-auto":
            from experiment_fasttext import ExperimentFasttext
            experiment = ExperimentFasttext(cfg)
            experiment.run_auto()
        elif cfg["module"] == "get_best_results":
            from asset_generators.get_best_results import get_best_results
            get_best_results()
        else:
            raise Exception("Unknown module!")
