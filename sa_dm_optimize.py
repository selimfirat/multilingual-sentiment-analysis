
def objective(trial):
    import numpy as np
    from experiment_deepmoji import ExperimentDeepMoji
    import argparse
    #import torch

    parser = argparse.ArgumentParser(description='DeepMoji Optuna')
    parser.add_argument("--device", type=str, default="cpu")
    
    params = {}

    params["embed_dropout_rate"] = 0.25
    params["final_dropout_rate"] = 0.5
    params["embed_l2"] = 1e-6

    params["initial_lr"] = 0.001
    params["next_lr"] = 0.0001

    """
    params["embed_dropout_rate"] = trial.suggest_uniform("embed_dropout_rate", 0.0, 0.7)
    params["final_dropout_rate"] = trial.suggest_uniform("final_dropout_rate", 0.0, 0.9)
    params["embed_l2"] = trial.suggest_loguniform("embed_l2", 1e-8, 1e-4)

    params["initial_lr"] = trial.suggest_loguniform("initial_lr", 1e-5, 1e-1)
    params["next_lr"] = trial.suggest_loguniform("next_lr", 1e-6, 1e-2)
    """

    cfg = vars(parser.parse_args())

    cfg["data"] = 'SemEval_English'
    #cfg["data"] = 'SE0714'#'Olympic'
    #cfg["device"] = '5, 6'
    cfg["dm_mode"] = "ft"
    cfg["pretrained"] = "xlm-roberta-large"

    params.update(cfg)
    np.random.seed(61)
    #torch.manual_seed(61)

    print("Experimenting with the data " + params['data'])
    experiment = ExperimentDeepMoji(params)
    val_result, test_result, model = experiment.val_metrics, experiment.test_metrics, experiment.model
    #torch.cuda.empty_cache()

    del experiment

    val_f1 = val_result["macro_f1"]
    print(val_result)
    print(val_f1)
    
    test_f1 = test_result["macro_f1"]
    print(test_result)
    print(test_f1)

    return val_f1

if __name__ == '__main__':
    objective(1)
"""
if __name__ == '__main__':
    import argparse
    import optuna
    from optuna.samplers import TPESampler, RandomSampler

    parser = argparse.ArgumentParser(description='Sentiment Deep Metric Learning')
    parser.add_argument("--data", default="SemEval_Arabic_English_Spanish", type=str)
    parser.add_argument("--study", type=str, default="dm_1")
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    cfg = vars(parser.parse_args())

    cfg["experiment"] = "optimize_" + cfg['data'] + "_" + cfg['study']
    cfg["pretrained"] = "xlm-roberta-large"

    global best_f1
    best_f1 = -1
    
    sampler = TPESampler(seed=61)
    study = optuna.create_study(study_name=cfg["study"], sampler=sampler, storage="sqlite:///figures/optimize_" + cfg['data'] + "_" + cfg['study'] + ".db", load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=100)
# python optimize.py --data emoji-tweets --study emoji-tweets-50k
"""
