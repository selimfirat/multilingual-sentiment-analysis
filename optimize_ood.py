
def objective(trial):
    import numpy as np
    from experiment_ml import ExperimentMultiLabelClassification
    import torch

    params = {}

    params["lr"] = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    params["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-8, 1e-4)
    params["dropout"] = trial.suggest_uniform("dropout", 0.0, 0.7)
    params["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 256])
    params["num_layers"] = trial.suggest_int("num_layers", 1, 2)
    params["weight_smoothing"] = trial.suggest_uniform("weight_smoothing", 0, 1.0)
    params["thresholding"] = "class_specific"
    params["loss"] = "focal"
    params["weights"] = "dynamic_loss_size"
    params["no_train"] = False
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    params["lambda"] = 0.003
    params["lr_center"] = 0.5
    params["gamma"] = trial.suggest_categorical("gamma", [0.5, 1.0, 2.0, 4.0])

    params.update(cfg)
    np.random.seed(61)
    torch.manual_seed(61)

    print(f"Experimenting with the data {params['data']}")
    experiment = ExperimentMultiLabelClassification(params)
    test_result = experiment.run()
    torch.cuda.empty_cache()

    del experiment

    return test_result["val_macro_f1"]


if __name__ == '__main__':
    import argparse
    import optuna
    from optuna.samplers import TPESampler, RandomSampler

    parser = argparse.ArgumentParser(description='Sentiment Deep Metric Learning')
    parser.add_argument("--data", default="SemEval_Arabic_English_Spanish", type=str)
    parser.add_argument("--study", type=str, default="ood_concat_final")
    parser.add_argument("--finetune", type=str, default="2178c82b9bc249cea3b485a83c3ecb5c")
    parser.add_argument("--device", type=str, default="cuda:0")

    cfg = vars(parser.parse_args())
    cfg["module"] = "ml"
    cfg["experiment"] = f"optimize_{cfg['data']}_{cfg['study']}"
    cfg["model"] = "lstm"
    cfg["num_epochs"] = 100
    cfg["patience"] = 5

    cfg["pooling"] = "attention"
    cfg["bidirectional"] = True
    #cfg["weights"] = None
    #cfg["thresholding"] = "class_specific"
    cfg["finetune_type"] = "concat_final" #"concat_hidden"
    cfg["pretrained"] = "xlm-roberta-large"
    cfg["prepro"] = "all"
    cfg["freeze_bert"] = False
    cfg["center_loss"] = False
    #cfg["loss"] = "bce"
    cfg["num_bins"] = 1000
    cfg["alpha"] = 1.0
    cfg["beta"] = 0.9999
    #cfg["dropout"] = 0
    cfg["rec_dropout"] = 0

    sampler = TPESampler(seed=61)
    study = optuna.create_study(study_name=cfg["study"], sampler=sampler, storage=f"sqlite:///figures/optimize_{cfg['data']}_{cfg['study']}.db", load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=100)
# MKL_THREADING_LAYER=GNU python optimize.py --data SemEval_Arabic_English --study fixed_Arabic_English --device cuda:0
# MKL_THREADING_LAYER=GNU python optimize.py --data SemEval_Arabic_Spanish --study fixed_Arabic_Spanish --device cuda:1
# MKL_THREADING_LAYER=GNU python optimize.py --data SemEval_English_Spanish --study fixed_English_Spanish --device cuda:2

# python optimize_ood.py --device cuda:0 --data SE0714
# python optimize_ood.py --device cuda:0 --data Olympic
# python optimize_ood.py --device cuda:0 --data PsychExp