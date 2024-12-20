import inspect
from itertools import permutations
import itertools
import json
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping
import time
import portalocker
import torch
import joblib
import pytz
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

# tz_str = "US/Eastern"
tz_str = "US/Pacific"
tz_ = pytz.timezone(tz_str)
os.environ['TZ'] = tz_str
time.tzset()

# default_loss_name = "mae"
default_loss_name = "medae"

# &&& param
results_base_path = "result_artifacts"
# results_base_path = "result_artifacts_temp"

# &&& param
# study_name = get_config_id(search_space)[:16]  # Half of actual length.
study_name = "study-17"

study_db_url = f"sqlite:///{results_base_path}/studies.db"
lock_path = Path(f"{results_base_path}/result_df.lock")
result_df_path = Path(f"{results_base_path}/result_df.parquet")

config_id_str = "config_id"

relevant_cols = ["config_id", "datetime_start", "duration", "medae", "mae", "mse", "study_name", "config"]


def get_inner_layer_permutations(hidden_edges):
    # Generate all permutations of lengths 1 to len(nums)
    all_permutations = [
        json.dumps(list(p)) for r in range(1, len(hidden_edges) + 1) for p in permutations(hidden_edges, r)
    ]
    return all_permutations


def get_config(trial):

    # TODO move these into arg module.

    # &&& param

    search_space = {  # big run

        "batch_size": trial.suggest_int("batch_size", 4, 8, step=2),

        "early_stop_patience": trial.suggest_int("early_stop_patience", 100, 100, step=50),

        # "early_stop_threshold": round(trial.suggest_float("early_stop_threshold", 0.00001, 0.00001), 5),
        "early_stop_threshold": trial.suggest_categorical("early_stop_threshold", [0.00001, 0.0001]),

        "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["median"]),

        "k_folds": trial.suggest_int("k_folds", 5, 5),

        "loss_name": trial.suggest_categorical("loss_name", [default_loss_name]),

        "lr_factor": round(trial.suggest_float("lr_factor", 0.1, 0.1, step=0.1), 1),  # 0.1 is the default.

        "lr_init": round(trial.suggest_float("lr_init", 0.001, 0.001, step=0.00001), 3),

        "lr_patience": trial.suggest_int("lr_patience", 25, 25),

        "lr_threshold": round(trial.suggest_float("lr_threshold", 0.001, 0.001, step=0.01), 3),

        "max_epochs": trial.suggest_int("max_epochs", 9999, 9999),

        "model.activation_func": trial.suggest_categorical("model.activation_func", [
            "elu", #"leakyrelu", "relu",
            # "celu", "elu", "gelu", "leakyrelu", "relu", "silu",
        ]),

        # Having `0.0, 0.0, step=0.001` will produce only `"model.dropout": 0.0,`.
        "model.dropout": round(trial.suggest_float("model.dropout", 0.0, 0.0, step=0.01), 2),

        "model.hidden_edges": trial.suggest_categorical("model.hidden_edges", [
            json.dumps([32, 16, 1, 2, 8, 4]),
            json.dumps([32, 16, 4, 8]),
            json.dumps([32]),
        ]),
        # "model.hidden_edges": trial.suggest_categorical("model.hidden_edges", get_inner_layer_permutations(
        #     [16, 32, 64, 128],
        # )),

        "model.input_dim": trial.suggest_int("model.input_dim", 1000, 1000),

        "model.model_class": trial.suggest_categorical("model.model_class", ["DeepMAgeModel"]),

        "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site"]),

        "predictor_class": trial.suggest_categorical("predictor_class", ["DeepMAgePredictor"]),

        # The larger the number, the more number of samples will be imputed, instead of filtered out.
        "remove_nan_samples_perc_2": trial.suggest_int("remove_nan_samples_perc_2", 80, 80, step=10),

        "split_train_test_by_percent": trial.suggest_categorical("split_train_test_by_percent", [False]),

        "test_ratio": trial.suggest_float("test_ratio", 0.2, 0.2),

        "weight_decay": round(trial.suggest_float("weight_decay", 0.000013, 0.000013, step=0.000001), 6),

    }

    # search_space = {  # benchmark run
    #
    #     "batch_size": trial.suggest_int("batch_size", 32, 32),
    #     # "batch_size": trial.suggest_int("batch_size", 32, 64, step=32),
    #     # "batch_size": trial.suggest_int("batch_size", 64, 64),
    #
    #     "early_stop_patience": trial.suggest_int("early_stop_patience", 30, 30),
    #
    #     "early_stop_threshold": trial.suggest_float("early_stop_threshold", 0.001, 0.001),
    #
    #     "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["median"]),
    #     # "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["mean"]),
    #
    #     # "k_folds": trial.suggest_int("k_folds", 5, 5),
    #     "k_folds": trial.suggest_int("k_folds", 1, 1),
    #
    #     "loss_name": trial.suggest_categorical("loss_name", [default_loss_name]),
    #
    #     "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.5),
    #
    #     "lr_init": trial.suggest_float("lr_init", 0.001, 0.001),
    #
    #     "lr_patience": trial.suggest_int("lr_patience", 10, 10),
    #
    #     "lr_threshold": trial.suggest_float("lr_threshold", 0.001, 0.001),
    #
    #     "max_epochs": trial.suggest_int("max_epochs", 3, 3),
    #     # "max_epochs": trial.suggest_int("max_epochs", 999, 999),
    #
    #     "model.activation_func": trial.suggest_categorical("model.activation_func", ["elu", "relu"]),
    #
    #     "model.dropout": trial.suggest_float("model.dropout", 0.1, 0.3, step=0.2),
    #
    #     "model.hidden_edges": trial.suggest_categorical("model.hidden_edges", [
    #         json.dumps([
    #             trial.suggest_int("hl1", 512, 512),
    #             trial.suggest_int("hl2", 512, 512),
    #             trial.suggest_int("hl3", 256, 256),
    #             trial.suggest_int("hl4", 128, 128),
    #         ]),
    #     ]),
    #
    #     "model.input_dim": trial.suggest_int("model.input_dim", 1000, 1000),
    #
    #     "model.model_class": trial.suggest_categorical("model.model_class", ["DeepMAgeModel"]),
    #
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_study_per_site"]),
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site", "per_study_per_site"]),
    #     "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site"]),
    #
    #     "predictor_class": trial.suggest_categorical("predictor_class", ["DeepMAgePredictor"]),
    #
    #     "remove_nan_samples_perc_2": trial.suggest_int("remove_nan_samples_perc_2", 20, 20, step=20),
    #
    #     "split_train_test_by_percent": trial.suggest_categorical("split_train_test_by_percent", [False]),
    #
    #     "test_ratio": trial.suggest_float("test_ratio", 0.2, 0.2, step=0.1),
    #
    #     "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.001),
    #
    # }

    # search_space = {  # dev run
    #
    #     "batch_size": trial.suggest_int("batch_size", 4, 8, step=2),
    #     # "batch_size": trial.suggest_int("batch_size", 32, 64, step=32),
    #     # "batch_size": trial.suggest_int("batch_size", 64, 64),
    #
    #     "early_stop_patience": trial.suggest_int("early_stop_patience", 30, 30),
    #
    #     "early_stop_threshold": trial.suggest_float("early_stop_threshold", 0.001, 0.001),
    #
    #     "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["median"]),
    #     # "imputation_strategy": trial.suggest_categorical("imputation_strategy", ["mean"]),
    #
    #     # "k_folds": trial.suggest_int("k_folds", 5, 5),
    #     "k_folds": trial.suggest_int("k_folds", 1, 1),
    #
    #     "loss_name": trial.suggest_categorical("loss_name", [default_loss_name]),
    #
    #     "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.5),
    #
    #     "lr_init": trial.suggest_float("lr_init", 0.001, 0.001),
    #
    #     "lr_patience": trial.suggest_int("lr_patience", 10, 10),
    #
    #     "lr_threshold": trial.suggest_float("lr_threshold", 0.001, 0.001),
    #
    #     "max_epochs": trial.suggest_int("max_epochs", 3, 3),
    #     # "max_epochs": trial.suggest_int("max_epochs", 999, 999),
    #
    #     "model.activation_func": trial.suggest_categorical("model.activation_func", ["elu", "relu"]),
    #
    #     "model.dropout": trial.suggest_float("model.dropout", 0.1, 0.3, step=0.2),
    #
    #     "model.hidden_edges": trial.suggest_categorical("model.hidden_edges", [
    #         json.dumps([512, 512, 256, 128]),
    #     ]),
    #
    #     "model.input_dim": trial.suggest_int("model.input_dim", 1000, 1000),
    #
    #     "model.model_class": trial.suggest_categorical("model.model_class", ["DeepMAgeModel"]),
    #
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_study_per_site"]),
    #     # "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site", "per_study_per_site"]),
    #     "normalization_strategy": trial.suggest_categorical("normalization_strategy", ["per_site"]),
    #
    #     "predictor_class": trial.suggest_categorical("predictor_class", ["DeepMAgePredictor"]),
    #
    #     "remove_nan_samples_perc_2": trial.suggest_int("remove_nan_samples_perc_2", 20, 20, step=20),
    #
    #     "split_train_test_by_percent": trial.suggest_categorical("split_train_test_by_percent", [False]),
    #
    #     "test_ratio": trial.suggest_float("test_ratio", 0.2, 0.2, step=0.1),
    #
    #     "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.001),
    #
    # }

    return search_space

# TODO for safekeeping
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

    ## &&& param

    n_trials = 20
    # n_trials = 40

    n_startup_trials = n_trials  # This effectively disables pruning.
    # n_startup_trials = 20

    n_min_trials = n_startup_trials

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=20,
        n_min_trials=n_min_trials,
    )

    # search_space: Mapping[str, Sequence[GridValueType]] = {}
    # sampler = GridSampler(search_space, seed=42) # TODO is there a way to fix this?
    sampler = TPESampler()

    print(f"Using {sampler.__class__.__name__}.")

    if restart and study_name in optuna.study.get_all_study_names(storage=study_db_url):
        optuna.delete_study(study_name=study_name, storage=study_db_url)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        storage=study_db_url,
        load_if_exists=True,
    )

    print_study_counts(study, sampler)

    fjdkfjd = 1

    # TODO do these two functions need to be here inside now?
    def objective(trial):

        # &&& param
        config = get_config(trial)
        # config = {
        #     "batch_size": 4,
        #     "early_stop_patience": 100,
        #     "early_stop_threshold": 0.0001,
        #     "imputation_strategy": "median",
        #     "k_folds": 5,
        #     "loss_name": "medae",
        #     "lr_factor": 0.1,
        #     "lr_init": 0.001,
        #     "lr_patience": 25,
        #     "lr_threshold": 0.001,
        #     "max_epochs": 4,
        #     "model.activation_func": "elu",
        #     "model.dropout": 0.0,
        #     "model.hidden_edges": "[32, 16, 1, 2, 8, 4]",
        #     "model.input_dim": 1000,
        #     "model.model_class": "DeepMAgeModel",
        #     "normalization_strategy": "per_site",
        #     "predictor_class": "DeepMAgePredictor",
        #     "remove_nan_samples_perc_2": 80,
        #     "split_train_test_by_percent": False,
        #     "test_ratio": 0.2,
        #     "weight_decay": 1.3e-05
        # }

        set_seeds()  # Reset seeds for reproducibility
        config_id = get_config_id(config)

        print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")
        print(f"config:\n{json.dumps(config, indent=4)}")
        with portalocker.Lock(lock_path, mode="a", timeout=10):
            func_name = inspect.currentframe().f_code.co_name
            print(f"Lock acquired in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")
            result_df = get_result_df()
            print(f"Lock released in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")

        if not override and not result_df.empty and config_id in set(result_df["config_id"].to_list()):
            print(f"Found existing results.")
            results = result_df[result_df["config_id"] == config_id].to_dict(orient="records")
            if len(results) > 1:
                # TODO Remove this check. we are already checking in callback.
                print(
                    f"WARNING: objective: "
                    f"Duplicate rows with different '{default_loss_name}' values found for config_id: {config_id}"
                )
                for result in results:
                    print(f"\t{default_loss_name}: {result[default_loss_name]}")
                result_dict = min(
                    results, key=lambda x: x[default_loss_name]
                    if isinstance(x[default_loss_name], float) else
                    float("inf")
                )
            else:
                result_dict = results[0]
        else:
            print(f"Running a new training pipeline.")
            dt_ = datetime.now()
            # print(f"config:\n{json.dumps(config, indent=4)}")
            predictor_class_name = config["predictor_class"]
            # noinspection PyTypeChecker
            predictor_class = globals()[predictor_class_name]
            predictor = predictor_class(config)
            result_dict = predictor.train_pipeline(trial)
            print(f"Total runtime for config_id '{config_id}': {datetime.now() - dt_}")

        # Attach custom attributes
        trial.set_user_attr("config", json.dumps(config))
        trial.set_user_attr("config_id", config_id)
        trial.set_user_attr("study_name", trial.study.study_name)
        trial.set_user_attr("mae", result_dict["mae"])
        trial.set_user_attr("medae", result_dict["medae"])
        trial.set_user_attr("mse", result_dict["mse"])

        loss = result_dict[config["loss_name"]]  # Minimize the loss
        return loss

    # noinspection PyUnusedLocal
    ## trial
    # noinspection PyShadowingNames
    ## study
    # TODO can I move all the df code at the end of the objective func?
    def save_study_callback(study, trial):

        curr_df = study.trials_dataframe()

        # Remove prefix 'user_attrs_' from any columns that have it.
        cols_dict = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
        curr_df = curr_df.rename(columns=cols_dict)

        # # TODO mod values, if needed.
        # index_to_update = curr_df[curr_df[config_id_str] == "b7c65ed3969908def6261778cc2a4c2a"].index[0]
        # curr_df.loc[index_to_update, default_loss_name] = 16.955668

        with portalocker.Lock(lock_path, mode="a", timeout=10):
            func_name = inspect.currentframe().f_code.co_name
            print(f"Lock acquired in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")
            result_df = get_result_df()

            result_df = pd.concat([result_df, curr_df], ignore_index=True)

            # Bring all the relevant columns in the front.
            # cols = relevant_cols
            cols = relevant_cols + sorted(set(result_df.columns) - set(relevant_cols))
            result_df = result_df[cols]

            # # check for duplications
            # duplicated_df = result_df[result_df.duplicated(subset=[config_id_str], keep=False)]
            # duplicated_df = duplicated_df.sort_values(by=[config_id_str])
            # print(f"AAA duplicated_df 1:\n{duplicated_df.drop(columns=['config'])}")

            # Deduplicate based on config_id and similar 'default_loss_name' values, but without modifying the original precision.
            default_loss_name_rounded = f'{default_loss_name}_rounded'
            result_df[default_loss_name_rounded] = result_df[default_loss_name].round(6)
            result_df = result_df.drop_duplicates(subset=[config_id_str, default_loss_name_rounded], keep="last", ignore_index=True)
            result_df = result_df.drop(columns=[default_loss_name_rounded])

            # check for duplications
            duplicated_df = result_df[result_df.duplicated(subset=[config_id_str], keep=False)]
            if len(duplicated_df) != 0:
                duplicated_df = duplicated_df.sort_values(by=[config_id_str])
                print(f"WARNING: save_study_callback: duplicated df:\n{duplicated_df.drop(columns=['config'])}")

            result_df = result_df.sort_values(by=default_loss_name)

            result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
            print(f"Saved result_df to: {result_df_path}")
            print(f"Lock released in '{func_name}'. Process name: '{getproctitle()}'. Pid: '{os.getpid()}'.")

        print(f"result_df:\n{result_df[relevant_cols].drop(columns=['config', 'study_name'])}")

        # no need for this anymore.
        # joblib.dump(study, study_path)
        # print(f"Saved study '{study.study_name}' to: {study_path}")

        print_study_counts(study, sampler)
        print(f"Now dt: {datetime.now(tz_)}")

    # noinspection PyUnresolvedReferences
    if isinstance(sampler, TPESampler) or not sampler.is_exhausted(study):
        study.optimize(
            objective,
            n_trials=n_trials,
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
    #   TODO maybe get that back in, once we move to true multiprocessing.
    # restart - restart a study.
    main(override=True, restart=False)
