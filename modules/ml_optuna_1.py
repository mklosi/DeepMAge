import json
from datetime import datetime
from pathlib import Path

import joblib
from hyperopt import hp
import pandas as pd
import optuna
from hyperopt import Trials
from optuna.samplers import GridSampler, TPESampler
from torch.onnx.symbolic_opset16 import grid_sampler

from modules.memory import Memory
from modules.ml_common import merge_dicts, get_config_id
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor, set_seeds

from hyperopt import fmin, tpe, STATUS_OK
from modules.ml_common import merge_dicts, get_config_id

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

results_base_path = "result_artifacts"  # &&& this should be somewhere else.
result_df_path = Path(f"{results_base_path}/result_df_optuna_1.parquet")
study_name = f"study_optuna_1"
study_path = Path(f"{results_base_path}/{study_name}.pkl")

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

    # "max_epochs": [4],
    "max_epochs": [2, 3],

    # "batch_size": [32],
    "batch_size": [32, 64],
    # "batch_size": [64, 32],
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
    config_id = get_config_id(config)

    set_seeds()  # Reset seeds for reproducibility

    print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")
    # print(f"config:\n{json.dumps(config, indent=4)}")
    predictor_class_name = config["predictor_class"]
    predictor_class = globals()[predictor_class_name]
    predictor = predictor_class(config)

    result_dict = predictor.train_pipeline()

    # Attach custom attributes
    trial.set_user_attr("config", config)
    trial.set_user_attr("config_id", config_id)
    trial.set_user_attr("mae", result_dict["mae"])
    trial.set_user_attr("medae", result_dict["medae"])
    trial.set_user_attr("mse", result_dict["mse"])

    loss = result_dict[config["loss_name"]]  # Minimize the loss
    return loss


# noinspection PyUnusedLocal
def save_study_callback(study, trial):

    result_df = study.trials_dataframe()

    # Remove prefix 'user_attrs_' from any columns that have it.
    cols = {col: col.replace('user_attrs_', '') for col in result_df.columns}
    result_df = result_df.rename(columns=cols)

    # Bring all the relevant columns in the front.
    relevant_cols = ["config_id", "datetime_start", "duration", "mse", "mae", "medae", "config"]
    # cols = relevant_cols + sorted(set(result_df.columns) - set(relevant_cols))
    result_df = result_df[relevant_cols]

    # Deduplicate based on config_id and latest trail.
    result_df = result_df.loc[result_df.groupby('config_id')['datetime_start'].idxmax()]
    result_df = result_df.reset_index(drop=True).set_index("config_id")

    # Sort and print.
    result_df = result_df.sort_values(by=default_loss_name)
    print(f"result_df:\n{result_df.drop(columns='config')}")

    result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
    print(f"Saved result_df to: {result_df_path}")

    joblib.dump(study, study_path)
    print(f"Saved study '{study.study_name}' to: {study_path}")


def main(overwrite):

    mem = Memory(noop=False)
    start_dt = datetime.now()

    if study_path.exists() and not overwrite:
        study = joblib.load(study_path)
        sampler = study.sampler
    else:
        # sampler = GridSampler(search_space, seed=42)
        sampler = TPESampler(seed=42)

        study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler)

    if not isinstance(sampler, TPESampler):
        # noinspection PyProtectedMember
        config_count = len(sampler._all_grids)
        print(f"Found '{config_count}' total configs.")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = config_count - completed_trials
        print(f"Running '{remaining_trials}' new train pipelines...")

        # assert sampler.is_exhausted(study) == (completed_trials == config_count) # &&&

    if isinstance(sampler, TPESampler) or not sampler.is_exhausted(study):
        study.optimize(objective, n_trials=None, timeout=3600, callbacks=[save_study_callback])

    # Get the best trial
    best_trial = study.best_trial  # &&& do we need this?
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print(f"  Params: {best_trial.params}")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main(overwrite=False)
