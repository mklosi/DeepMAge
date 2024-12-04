import json
from datetime import datetime
from pathlib import Path
from hyperopt import hp
import pandas as pd
import optuna
from hyperopt import Trials
from modules.memory import Memory
from modules.ml_common import merge_dicts, get_config_id
from modules.ml_hyperopt import search_space
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor, set_seeds

from hyperopt import fmin, tpe, STATUS_OK
from modules.ml_common import merge_dicts, get_config_id

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# default_loss_name = "mae"
default_loss_name = "medae"

default_args_dict = {
    "predictor_class": "DeepMAgePredictor",
    "model.model_class": "DeepMAgeModel",
    "model.input_dim": 1000,
    "loss_name": default_loss_name,
}

search_space = {
    "imputation_strategy": ["median"],
    "max_epochs": [2, 3],
    "batch_size": [32, 64],
    "lr_init": [0.0001],
    "weight_decay": [0.0],
    "lr_factor": [0.1],
    "lr_patience": [10],
    "lr_threshold": [0.01],
    "early_stop_patience": [20],
    "early_stop_threshold": [0.0001],
    "model.inner_layers": [json.dumps([512, 512, 256, 128])],
    "model.dropout": [0.3],
    "model.activation_func": ["elu"],
    "remove_nan_samples_perc": [10],
    "test_ratio": [0.2],
}


def objective(trial):

    hyperparams = {param: trial.suggest_categorical(param, choices) for param, choices in search_space.items()}

    config = merge_dicts(default_args_dict, hyperparams)

    # Initialize the predictor
    set_seeds()  # Reset seeds for reproducibility
    predictor_class_name = config["predictor_class"]
    predictor_class = globals()[predictor_class_name]
    predictor = predictor_class(config)

    # Run training pipeline
    result = predictor.train_pipeline()

    loss = result[config["loss_name"]]  # Minimize the loss
    return loss


def main():

    mem = Memory(noop=False)
    start_dt = datetime.now()

    result_df_path = Path(f"result_artifacts/result_df_optuna_1.parquet")

    sampler = optuna.samplers.GridSampler(search_space, seed=42)
    study = optuna.create_study(study_name="optuna-study-1", direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=None, timeout=3600)

    # Get the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print(f"  Params: {best_trial.params}")

    result_df = study.trials_dataframe()

    result_df = result_df.sort_values(by="value")
    print(f"result_df:\n{result_df}")

    result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
    print(f"Saved result_df to: {result_df_path}")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
