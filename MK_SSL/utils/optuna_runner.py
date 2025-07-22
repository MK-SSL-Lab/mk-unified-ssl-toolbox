import optuna
from copy import deepcopy

def optimize_hyperparameters(
    trainer,
    train_dataset,
    val_dataset=None,
    n_trials=20,
    epochs=5,
    objective_metric="val_loss",
    search_space=None,
    pruner: str = "median",  
):
    """
    Runs Optuna hyperparameter search using the provided Trainer object.

    Args:
        trainer (Trainer): The Trainer object with a train() method.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset, optional): Validation dataset.
        n_trials (int): Number of Optuna trials.
        epochs (int): Max epochs during tuning phase.
        objective_metric (str): Metric name to minimize.
        search_space (dict, optional): Custom Optuna search space.
        pruner (str): Pruner strategy name. One of ["median", "successive_halving", "none"].

    Returns:
        dict: Best hyperparameters found.
    """

    search_space = search_space or {
        "lr": lambda trial: trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": lambda trial: trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": lambda trial: trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
    }

    def objective(trial):
        trial_trainer = deepcopy(trainer)
        hparams = {k: suggest_fn(trial) for k, suggest_fn in search_space.items()}
        trial_trainer._optuna_trial = trial

        trial_trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            use_optuna=False,
            use_embedding_logger=False,

            **hparams,
        )

        if hasattr(trial_trainer, "_optuna_trial"):
            del trial_trainer._optuna_trial

        return getattr(trial_trainer, f"last_{objective_metric}", float("inf"))

    # Select pruner
    pruner = pruner.lower()
    if pruner == "median":
        optuna_pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    elif pruner == "successive_halving":
        optuna_pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner == "none":
        optuna_pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unsupported pruner: {pruner}. Choose from 'median', 'successive_halving', or 'none'.")

    study = optuna.create_study(direction="minimize", pruner=optuna_pruner)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
