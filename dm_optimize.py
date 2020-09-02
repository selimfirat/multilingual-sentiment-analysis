
def objective(trial):
    import os
    import numpy as np
    np.random.seed(61)
    
    params = {}
    
    params["embed_dropout_rate"] = trial.suggest_uniform("embed_dropout_rate", 0.0, 0.7)
    params["final_dropout_rate"] = trial.suggest_uniform("final_dropout_rate", 0.0, 0.9)
    params["embed_l2"] = trial.suggest_loguniform("embed_l2", 1e-8, 1e-4)

    params["initial_lr"] = trial.suggest_loguniform("initial_lr", 1e-5, 1e-1)
    params["next_lr"] = trial.suggest_loguniform("next_lr", 1e-6, 1e-2)
    
    #cfg = self.cfg
    params["data"] = "SE0714"

    print("Experimenting with the data " + params['data'])
    
    param_str = "--embed_dropout_rate {} --final_dropout_rate {} --embed_l2 {} --initial_lr {} --next_lr {}".format(params['embed_dropout_rate'], params['final_dropout_rate'], params['embed_l2'], params['initial_lr'], params['next_lr'])
    
    res = os.system("~/anaconda3/envs/batu-dm/bin/python experiment_deepmoji.py --data SE0714 --verbose True " + param_str)
    
    print(res)
    
    """
    import json
    val_f1 = None
    with open('ex_dm_val_out.txt', 'w') as file:
        val_metric_dict = json.load(file)
        val_f1 = val_metric_dict["macro_f1"]
        print("trial result: ", val_f1)
        return val_f1
    """
    
    #experiment = ExperimentDeepMoji(params)
    #val_result, test_result, model = experiment.val_metrics, experiment.test_metrics, experiment.model

    #del experiment

    """
    val_f1 = val_result["macro_f1"]
    print(val_result)
    print(val_f1)
    
    test_f1 = test_result["macro_f1"]
    print(test_result)
    print(test_f1)

    return val_f1
    """

"""
if __name__ == '__main__':
    import argparse
    import optuna
    from optuna.samplers import TPESampler, RandomSampler

    parser = argparse.ArgumentParser(description='Sentiment Deep Metric Learning')
    parser.add_argument("--data", default="SemEval_English", type=str)
    parser.add_argument("--study", type=str, default="test_se")
    parser.add_argument("--device", type=str, default="6")

    cfg = vars(parser.parse_args())

    cfg["experiment"] = "optimize_" + cfg['data'] + "_" + cfg['study']
    cfg["pretrained"] = "xlm-roberta-large"

    global best_f1
    best_f1 = -1
    
    sampler = TPESampler(seed=61)
    study = optuna.create_study(study_name=cfg["study"], sampler=sampler, storage="sqlite:///" + cfg['study'] + ".db", load_if_exists=True, direction="maximize")
    self.cfg = cfg
    study.optimize(objective, n_trials=3)
    
"""

