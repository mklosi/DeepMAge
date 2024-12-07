import inspect
import itertools
import json
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping

import portalocker
import torch
import joblib
import optuna
import pandas as pd
from numpy.random.mtrand import Sequence
# noinspection PyProtectedMember
from optuna.samplers._grid import GridValueType
from optuna.trial import TrialState
from optuna import load_study
from optuna.samplers import GridSampler, TPESampler
from setproctitle import setproctitle, getproctitle

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
results_base_path = "result_artifacts"
# results_base_path = "result_artifacts_temp"

study_db_url = f"sqlite:///{results_base_path}/studies.db"
lock_path = Path(f"{results_base_path}/result_df.lock")
result_df_path = Path(f"{results_base_path}/result_df.parquet")


def get_config(trial):

    # &&& move these into arg module.

    # &&& param

    search_space = {
        "predictor_class": trial.suggest_categorical("predictor_class", ["DeepMAgePredictor"]),

        "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["mean", "median"]),

        "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site", "per_study_per_site"]),

        "split_train_test_by_percent": trial.suggest_categorical("split_train_test_by_percent", [False, True]),

        "max_epochs": trial.suggest_int("max_epochs", 999, 999),

        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024]),

        "lr_init": trial.suggest_float("lr_init", 0.0001, 0.1),

        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.5),

        "lr_factor": trial.suggest_float("lr_factor", 0.1, 0.5),

        "lr_patience": trial.suggest_int("lr_patience", 10, 10),

        "lr_threshold": trial.suggest_float("lr_threshold", 0.001, 1.0),

        "early_stop_patience": trial.suggest_int("early_stop_patience", 50, 50),

        "early_stop_threshold": trial.suggest_float("early_stop_threshold", 0.001, 1.0),

        "model.model_class": trial.suggest_categorical("model.model_class", ["DeepMAgeModel"]),

        "model.input_dim": trial.suggest_int("model.input_dim", 1000, 1000),

        "model.inner_layers": trial.suggest_categorical("model.inner_layers", [
            json.dumps([512, 512, 256, 128]),
            json.dumps([512, 512, 512, 512, 512]),
            json.dumps([512, 512, 256, 256, 128, 64]),
            json.dumps([1024, 512, 256, 128, 64, 32, 16, 8]),
        ]),

        "model.dropout": trial.suggest_float("model.dropout", 0.1, 0.5),

        "model.activation_func": trial.suggest_categorical("model.activation_func", ["elu", "relu"]),

        "remove_nan_samples_perc": trial.suggest_int("remove_nan_samples_perc", 10, 30, step=10),

        "test_ratio": trial.suggest_float("test_ratio", 0.2, 0.2),

        "loss_name": trial.suggest_categorical("loss_name", [default_loss_name]),
    }

    # search_space = {
    #     "predictor_class": trial.suggest_categorical("predictor_class", ["DeepMAgePredictor"]),
    #
    #     "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["median"]),
    #
    #     # "some_other_hyperparameter_4": [40], # testing hyperparameter.
    #     # "some_other_hyperparameter_5": [50], # testing hyperparameter.
    #
    #     "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_study_per_site"]),
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site"]),
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_study_per_site", "per_site"]),
    #
    #     "split_train_test_by_percent": trial.suggest_categorical("split_train_test_by_percent", [False]),
    #
    #     # "max_epochs": trial.suggest_int("max_epochs", [20]),
    #     "max_epochs": trial.suggest_int("max_epochs", 2, 3),  # <--
    #     # "max_epochs": trial.suggest_int("max_epochs", [3, 2]),
    #
    #     # "batch_size": trial.suggest_int("batch_size", [32]),
    #     "batch_size": trial.suggest_int("batch_size", 32, 64, step=32),  # <--
    #     # "batch_size": trial.suggest_int("batch_size", [32, 64]),  # <--
    #     # "batch_size": trial.suggest_int("batch_size", [64, 32]),
    #     # "batch_size": trial.suggest_int("batch_size", [32, 64, 128, 256]),
    #
    #     "lr_init": trial.suggest_float("lr_init", 0.0001, 0.0001),
    #
    #     "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.0),
    #
    #     "lr_factor": trial.suggest_float("lr_factor", 0.1, 0.1),
    #
    #     "lr_patience": trial.suggest_int("lr_patience", 10, 10),
    #
    #     "lr_threshold": trial.suggest_float("lr_threshold", 0.01, 0.01),
    #
    #     "early_stop_patience": trial.suggest_int("early_stop_patience", 20, 20),
    #
    #     "early_stop_threshold": trial.suggest_float("early_stop_threshold", 0.0001, 0.0001),
    #
    #     "model.model_class": trial.suggest_categorical("model.model_class", ["DeepMAgeModel"]),
    #
    #     "model.input_dim": trial.suggest_int("model.input_dim", 1000, 1000),
    #
    #     # &&& get more creative here. also fix number of actual hidden layers are -1.
    #     "model.inner_layers": trial.suggest_categorical("model.inner_layers", [
    #         json.dumps([
    #             trial.suggest_int("hl1", 512, 512),
    #             trial.suggest_int("hl2", 512, 512),
    #             trial.suggest_int("hl3", 256, 256),
    #             trial.suggest_int("hl4", 128, 128),
    #         ]),
    #         # more options here.
    #     ]),
    #
    #     "model.dropout": trial.suggest_float("model.dropout", 0.3, 0.3),
    #
    #     "model.activation_func": trial.suggest_categorical("model.activation_func", ["elu"]),
    #
    #     "remove_nan_samples_perc": trial.suggest_int("remove_nan_samples_perc", 10, 10),
    #
    #     "test_ratio": trial.suggest_float("test_ratio", 0.2, 0.2),
    #
    #     "loss_name": trial.suggest_categorical("loss_name", [default_loss_name]),
    # }

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

    return search_space

# &&& for safekeeping
# search_space = {key: search_space[key] for key in sorted(search_space)}  # This is needed to correctly get ids.
# search_space = {key: sorted(search_space[key]) for key in sorted(search_space)}  # This is needed to correctly get ids.
# search_space = {key: sorted([search_space[key]]) for key in sorted(search_space)}  # This is needed to correctly get ids.


def print_study_counts(study, sampler=None):
    if sampler is not None and isinstance(sampler, GridSampler):
        # noinspection PyProtectedMember
        config_count = len(sampler._all_grids)
        print(f"config_count: {config_count}")

    total_trials = len(study.trials)
    complete_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    fail_trials = len([t for t in study.trials if t.state == TrialState.FAIL])
    pruned_trials = len([t for t in study.trials if t.state == TrialState.PRUNED])
    running_trials = len([t for t in study.trials if t.state == TrialState.RUNNING])
    waiting_trials = len([t for t in study.trials if t.state == TrialState.WAITING])
    print(
        f"total_trials: {total_trials}\n"
        f"completed_trials: {complete_trials}\n"
        f"fail_trials: {fail_trials}\n"
        f"pruned_trials: {pruned_trials}\n"
        f"running_trials: {running_trials}\n"
        f"waiting_trials: {waiting_trials}"
    )

    # &&&
    sum_ = complete_trials + fail_trials + pruned_trials + running_trials + waiting_trials
    # assert sum_ == total_trials
    if sum_ != total_trials:
        print(f"WARNING: sum_: {sum_}, total_trials: {total_trials}.")


def get_result_df():
    if result_df_path.exists():
        print(f"Reading result_df from: {result_df_path}")
        result_df = pd.read_parquet(result_df_path)
    else:
        result_df = pd.DataFrame()
    return result_df


def main(override, restart):

    mem = Memory(noop=False)
    start_dt = datetime.now()

    # &&& param
    # study_name = get_config_id(search_space)[:16]  # Half of actual length.
    study_name = "study-2"

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

    # search_space: Mapping[str, Sequence[GridValueType]] = {}
    # sampler = GridSampler(search_space, seed=42) # &&& is there a way to fix this?

    sampler = TPESampler()

    print(f"Using {sampler.__class__.__name__}.")

    if restart and study_name in optuna.study.get_all_study_names(storage=study_db_url):
        optuna.delete_study(study_name=study_name, storage=study_db_url)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=study_db_url,
        load_if_exists=True,
    )

    print_study_counts(study, sampler)

    fjdkfjd = 1 # &&&

    # &&& do these two functions need to be here inside now?
    def objective(trial):

        set_seeds()  # Reset seeds for reproducibility

        config = get_config(trial)
        config = {key: config[key] for key in sorted(config)}
        config_id = get_config_id(config)

        print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")
        with portalocker.Lock(lock_path, mode="a", timeout=10):
            func_name = inspect.currentframe().f_code.co_name
            print(f"Lock acquired in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")
            result_df = get_result_df()
            print(f"Lock released in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")

        if not override and not result_df.empty and config_id in set(result_df["config_id"].to_list()):
            print(f"Found existing results.")
            results = result_df[result_df["config_id"] == config_id].to_dict(orient="records")
            assert len(results) == 1
            result_dict = results[0]
        else:
            print(f"Running a new training pipeline.")
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
    # &&& can I move all the df code at the end of the objective func?
    def save_study_callback(study, trial):

        curr_df = study.trials_dataframe()

        # Remove prefix 'user_attrs_' from any columns that have it.
        cols_dict = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
        curr_df = curr_df.rename(columns=cols_dict)

        # Bring all the relevant columns in the front.
        relevant_cols = ["config_id", "datetime_start", "duration", "mse", "mae", "medae", "config"]
        # cols = relevant_cols
        cols = relevant_cols + sorted(set(curr_df.columns) - set(relevant_cols))
        curr_df = curr_df[cols]

        with portalocker.Lock(lock_path, mode="a", timeout=10):
            func_name = inspect.currentframe().f_code.co_name
            print(f"Lock acquired in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")
            result_df = get_result_df()

            result_df = pd.concat([result_df, curr_df], ignore_index=True)
            result_df = result_df.loc[result_df.groupby("config_id")['datetime_start'].idxmax()]
            result_df = result_df.sort_values(by=default_loss_name)

            result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
            print(f"Saved result_df to: {result_df_path}")
            print(f"Lock released in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")

        print(f"result_df:\n{result_df[relevant_cols].drop(columns='config')}")

        # no need for this anymore.
        # joblib.dump(study, study_path)
        # print(f"Saved study '{study.study_name}' to: {study_path}")

        print_study_counts(study, sampler)

    # noinspection PyUnresolvedReferences
    if isinstance(sampler, TPESampler) or not sampler.is_exhausted(study):
        study.optimize(
            objective,
            # &&& param
            # n_trials=20,
            n_trials=1000,
            timeout=None,
            n_jobs=1,
            callbacks=[save_study_callback],
            # show_progress_bar=True,
        )
    else:
        print("All trial exhausted.")

    print("--- Final study_counts ---------------------------------------------------------------------------")
    print_study_counts(study, sampler)

    # Get the best trial
    best_trial = study.best_trial
    print(f"Best trial:\n  Params: {best_trial.params}\n  Value: {best_trial.value}")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":

    process_name = f"python{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}-optuna"
    setproctitle(process_name)
    print(f"Main process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")

    # &&& param
    # override - rerun trails even if they are in the result_df.
    # overwrite - set result_df to study.trials_dataframe() on each save callback.
    #   &&& maybe get that back in, once we move to true multiprocessing.
    # restart - restart a study.
    main(override=False, restart=False)
