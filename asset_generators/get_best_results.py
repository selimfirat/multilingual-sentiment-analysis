def get_best_results():
    import mlflow
    import numpy as np
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiments = client.list_experiments()

    for experiment in experiments:
        df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        if "metrics.val_f1" in df:
            score = float(df.iloc[df["metrics.val_f1"].idxmax()]["metrics.test_f1"])
            score = np.round(score*100, 2)
            print(experiment.name, score)
