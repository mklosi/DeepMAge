import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from io import StringIO

from optuna.importance import get_param_importances
from optuna.storages import RDBStorage
from optuna.trial import TrialState

from optuna.visualization import plot_param_importances, plot_slice

from memory import Memory
from modules.ml_common import get_config_id, get_df_with_cols
from modules.ml_optuna_1 import print_study_counts, relevant_cols, config_id_str
from modules.ml_pipeline import default_loss_name

# &&& param
results_base_path = "result_artifacts"
# results_base_path = "result_artifacts_temp"


if __name__ == '__main__':

    pd.set_option('display.max_columns', 999)
    pd.set_option('display.min_rows', 20)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # mem = Memory(noop=False)
    # mem.log_memory(print, "before____")

    # # Find out the best config so far.
    # df = df.sort_values(by=default_loss_name)
    # ser = df.iloc[0]
    # loss = ser[default_loss_name]
    # config = json.loads(ser["config"])

    dfs = []

    storage_urls = [
        # f"sqlite:///{results_base_path}/studies_2024-12-07.db",
        f"sqlite:///{results_base_path}/studies.db",
    ]

    for storage_url in storage_urls:
        print(f"=== storage_url: {storage_url} ===============================================")

        study_names = optuna.study.get_all_study_names(storage=storage_url)
        print(f"Study count: {len(study_names)}")

        for study_name in study_names:

            # &&& param
            if storage_url != f"sqlite:///{results_base_path}/studies.db" or study_name != "study-10":
                continue

            print(f"--- study_name: {study_name} ------------------------")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            print_study_counts(study)

            # # Mark stale trials as failed.
            # storage = RDBStorage(storage_url)
            # running_trials = [t for t in study.trials if t.state == TrialState.RUNNING]
            # if running_trials:
            #     print(f"Marking '{len(running_trials)}' running trials as failed.")
            #     for trial in running_trials:
            #         # noinspection PyProtectedMember
            #         storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
            #     print_study_counts(study)

            # # Re-enqueue failed trails.
            # failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
            # if failed_trials:
            #     print(f"Re-enqueuing '{len(failed_trials)}' failed trials.")
            #     for trial in failed_trials:
            #         study.enqueue_trial(trial.params)
            #     print_study_counts(study)

            fjdkfdjfd = 1

            ## Plot

            print("plot_param_importances...")
            fig = plot_param_importances(study)
            fig.show()

            print("plot_slice...")
            fig = plot_slice(study)
            fig.show()

            ## result_df

            curr_df = study.trials_dataframe()

            # Remove prefix 'user_attrs_' from any columns that have it.
            cols_dict = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
            curr_df = curr_df.rename(columns=cols_dict)

            # Bring all the relevant columns in the front.
            # cols = relevant_cols
            cols = relevant_cols + sorted(set(curr_df.columns) - set(relevant_cols))
            curr_df = get_df_with_cols(curr_df, cols)

            dfs.append(curr_df)

    result_df = pd.concat(dfs)

    # Deduplicate based on config_id and similar 'default_loss_name' values, but without modifying the original precision.
    default_loss_name_rounded = f'{default_loss_name}_rounded'
    result_df[default_loss_name_rounded] = result_df[default_loss_name].round(6)
    result_df = result_df.drop_duplicates(subset=[config_id_str, default_loss_name_rounded], keep="last", ignore_index=True)
    # result_df = result_df.groupby(by=[config_id_str, default_loss_name_rounded], group_keys=False).tail(1)
    result_df = result_df.drop(columns=[default_loss_name_rounded])

    result_df = result_df.sort_values(by=default_loss_name)

    # check for duplications
    result_df_deduped = result_df.drop_duplicates(subset=[config_id_str], keep="last", ignore_index=True)
    assert len(result_df) == len(result_df_deduped)

    print("--------------------------------------------------------------------")
    print(f"result_df:\n{get_df_with_cols(result_df, relevant_cols).drop(columns='config')}")
    # print(f"result_df:\n{result_df.drop(columns='config')}")

    # mem.log_memory(print, "after_save")
