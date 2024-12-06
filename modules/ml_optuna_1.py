import itertools
import json
import threading
from datetime import datetime
from pathlib import Path
import torch
import joblib
import optuna
import pandas as pd
from optuna import load_study
from optuna.samplers import GridSampler, TPESampler

from modules.memory import Memory
from modules.ml_common import get_config_id
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor, set_seeds

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# default_loss_name = "mae"
default_loss_name = "medae"

# &&& param
# results_base_path = "result_artifacts"
results_base_path = "result_artifacts_temp"

study_db_url = f"sqlite:///{results_base_path}/studies.db"

# &&& move these into arg module.

# &&& param

# search_space = {
#     "predictor_class": ["DeepMAgePredictor"],
#     "imputation_strategy": ["mean", "median"],
#     "normalization_strategy": ["per_site", "per_study_per_site"],
#     "split_train_test_by_percent": [False, True],
#     "max_epochs": [999],
#     "batch_size": [16, 32, 64, 128],
#     "lr_init": [0.001],
#     "weight_decay": [0.0, 0.001],
#     "lr_factor": [0.1, 0.5],
#     "lr_patience": [10],
#     "lr_threshold": [0.001, 0.1],
#     "early_stop_patience": [30],
#     "early_stop_threshold": [0.001, 0.1],
#     "model.model_class": ["DeepMAgeModel"],
#     "model.input_dim": [1000],
#     "model.inner_layers": [
#         json.dumps([512, 512, 256, 128]),
#         json.dumps([512, 512, 512, 512, 512]),
#         json.dumps([512, 512, 256, 256, 128, 64]),
#         json.dumps([1024, 512, 256, 128, 64, 32, 16, 8]),
#     ],
#     "model.dropout": [0.1, 0.3],
#     "model.activation_func": ["elu", "relu"],
#     "remove_nan_samples_perc": [10, 30],
#     "test_ratio": [0.2],
#     "loss_name": [default_loss_name],
# }

search_space = {
    "predictor_class": ["DeepMAgePredictor"],

    "imputation_strategy": ["median"],

    "normalization_strategy": ["per_study_per_site"],
    # "normalization_strategy": ["per_site"],

    "split_train_test_by_percent": [False],

    # "max_epochs": [3],
    "max_epochs": [2, 3],  # actually runs as [3, 2]  <--
    # "max_epochs": [3, 2],

    # "batch_size": [32],
    # "batch_size": [64],  # <--
    # "batch_size": [32, 64],  # <--
    # "batch_size": [64, 32],
    "batch_size": [32, 64, 128, 256],
    "lr_init": [0.0001],
    "weight_decay": [0.0],
    "lr_factor": [0.1],
    "lr_patience": [10],
    "lr_threshold": [0.01],
    "early_stop_patience": [20],
    "early_stop_threshold": [0.0001],
    "model.model_class": ["DeepMAgeModel"],
    "model.input_dim": [1000],
    "model.inner_layers": [json.dumps([512, 512, 256, 128])],
    "model.dropout": [0.3],
    "model.activation_func": ["elu"],
    "remove_nan_samples_perc": [10],
    "test_ratio": [0.2],
    "loss_name": [default_loss_name],
}

# search_space = {
#     "batch_size": 16,
#     "early_stop_patience": 100,
#     "early_stop_threshold": 0.001,
#     "imputation_strategy": "mean",
#     "loss_name": "medae",
#     "lr_factor": 0.5,
#     "lr_init": 0.001,
#     "lr_patience": 50,
#     "lr_threshold": 0.001,
#     "max_epochs": 999,
#     "model.activation_func": "elu",
#     "model.dropout": 0.1,
#     "model.inner_layers": "[512, 512, 256, 128]",
#     "model.input_dim": 1000,
#     "model.model_class": "DeepMAgeModel",
#     "normalization_strategy": "per_study_per_site",
#     "predictor_class": "DeepMAgePredictor",
#     "remove_nan_samples_perc": 10,
#     "split_train_test_by_percent": False,
#     "test_ratio": 0.2,
#     "weight_decay": 0.0
# }

# &&& param
search_space = {key: sorted(search_space[key]) for key in sorted(search_space)}  # This is needed to correctly get ids.
# search_space = {key: sorted([search_space[key]]) for key in sorted(search_space)}  # This is needed to correctly get ids.


def main(override, overwrite, restart):

    mem = Memory(noop=False)
    start_dt = datetime.now()

    result_df_path = Path(f"{results_base_path}/result_df.parquet")

    if result_df_path.exists() and not overwrite:
        result_df_ = pd.read_parquet(result_df_path)
    else:
        result_df_ = pd.DataFrame()

    study_name = get_config_id(search_space)[:16]  # Half of actual length.

    # study_path = Path(f"{results_base_path}/study_{study_name}.pkl")
    #
    # if study_path.exists() and not restart:
    #     study = joblib.load(study_path)
    #     sampler = study.sampler
    #     print(f"Loaded existing study with name: {study_name}")
    # else:
    #     sampler = GridSampler(search_space, seed=42)
    #     # sampler = TPESampler(seed=42, multivariate=True)
    #
    #     study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler)

    # &&& param
    # sampler = GridSampler(search_space)
    sampler = TPESampler()

    if restart and study_name in optuna.study.get_all_study_names(storage=study_db_url):
        optuna.delete_study(study_name=study_name, storage=study_db_url)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=study_db_url,
        load_if_exists=True,
    )

    if isinstance(sampler, GridSampler):
        print(f"Using GridSampler.")
        # noinspection PyProtectedMember
        config_count = len(sampler._all_grids)
        print(f"Found '{config_count}' total configs.")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = config_count - completed_trials
        print(f"Running '{remaining_trials}' new train pipelines...")

    def objective(trial):

        set_seeds()  # Reset seeds for reproducibility

        config = {param: trial.suggest_categorical(param, choices) for param, choices in search_space.items()}
        config_id = get_config_id(config)

        if config_id in result_df_.index and not override:
            print(f"--- Found result for config_id: {config_id} --------------------------------------------")
            result_dict = result_df_.loc[config_id].to_dict()
        else:
            print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")
            # print(f"config:\n{json.dumps(config, indent=4)}")
            predictor_class_name = config["predictor_class"]
            # noinspection PyTypeChecker
            predictor_class = globals()[predictor_class_name]
            predictor = predictor_class(config)
            result_dict = predictor.train_pipeline()

        # Attach custom attributes
        trial.set_user_attr("config", json.dumps(config))
        trial.set_user_attr("config_id", config_id)
        trial.set_user_attr("mae", result_dict["mae"])
        trial.set_user_attr("medae", result_dict["medae"])
        trial.set_user_attr("mse", result_dict["mse"])

        loss = result_dict[config["loss_name"]]  # Minimize the loss
        return loss

    # noinspection PyUnusedLocal
    ## trial
    # noinspection PyShadowingNames
    ## study
    def save_study_callback(study, trial):

        curr_df = study.trials_dataframe()

        # Remove prefix 'user_attrs_' from any columns that have it.
        cols = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
        curr_df = curr_df.rename(columns=cols)

        # Bring all the relevant columns in the front.
        relevant_cols = ["config_id", "datetime_start", "duration", "mse", "mae", "medae", "config"]
        # cols = relevant_cols + sorted(set(result_df.columns) - set(relevant_cols))
        curr_df = curr_df[relevant_cols]

        # &&& do we need to fuck around with index here?  just make sure config_id is at the front.
        nonlocal result_df_
        result_df = pd.concat([result_df_, curr_df.set_index("config_id")])
        result_df = result_df.reset_index()
        result_df = result_df.loc[result_df.groupby("config_id")['datetime_start'].idxmax()]
        result_df = result_df.set_index("config_id")
        result_df = result_df.sort_values(by=default_loss_name)
        result_df_ = result_df

        print(f"result_df:\n{result_df.drop(columns='config')}")

        result_df.to_parquet(result_df_path, engine='pyarrow', index=True)
        print(f"Saved result_df to: {result_df_path}")

        # &&& no need for this anymore.
        # joblib.dump(study, study_path)
        # print(f"Saved study '{study.study_name}' to: {study_path}")

    if isinstance(sampler, TPESampler) or not sampler.is_exhausted(study):
        study.optimize(
            objective,
            n_trials=8,  # &&& param
            timeout=None,
            n_jobs=1,
            callbacks=[save_study_callback],
            # show_progress_bar=True,
        )

    # Get the best trial
    best_trial = study.best_trial
    print(f"Best trial:\n  Params: {best_trial.params}\n  Value: {best_trial.value}")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    # &&& param
    # override - rerun trails even if they are in the result_df.
    # overwrite - set result_df to study.trials_dataframe() on each save callback.
    # restart - restart a study.
    main(override=False, overwrite=False, restart=False)
