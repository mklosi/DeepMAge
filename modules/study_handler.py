import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from io import StringIO

from optuna.importance import get_param_importances
from optuna.trial import TrialState

from optuna.visualization import plot_param_importances, plot_slice

from memory import Memory
from modules.ml_common import get_config_id
from modules.ml_optuna_1 import print_study_counts
from modules.ml_pipeline import default_loss_name

# &&& param
results_base_path = "result_artifacts"
# results_base_path = "result_artifacts_temp"

study_db_url = f"sqlite:///{results_base_path}/studies.db"

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

    relevant_cols = ["config_id", "datetime_start", "duration", "mse", "mae", "medae", "config"]
    dfs = []

    study_summaries = optuna.study.get_all_study_summaries(storage=study_db_url)
    print(f"Study count: {len(study_summaries)}")
    for study_summary in study_summaries:
        print(f"--- study_name: {study_summary.study_name} ----------------------------------------------")
        study = optuna.load_study(study_name=study_summary.study_name, storage=study_db_url)
        print_study_counts(study)

        # # &&& param
        # failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]
        # if failed_trials:
        #     print(f"Re-enqueuing '{len(failed_trials)}' failed trials.")
        #     for trial in failed_trials:
        #         study.enqueue_trial(trial.params)
        #     print_study_counts(study)

        # print("param_importances:")
        # param_importances = get_param_importances(study)
        # for param, importance in param_importances.items():
        #     print(f"{param}: {importance:,.6f}")

        print("param_importances visualization...")
        fig = plot_param_importances(study)
        fig.show()

        # print("plot_parallel_coordinate visualization...")
        # fig = visualization.plot_parallel_coordinate(study)
        # fig.show()

        # print("plot_contour visualization...")
        # fig = visualization.plot_contour(study)
        # fig.show()

        print("plot_slice visualization...")
        fig = plot_slice(study)
        fig.show()

        curr_df = study.trials_dataframe()

        # Remove prefix 'user_attrs_' from any columns that have it.
        cols_dict = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
        curr_df = curr_df.rename(columns=cols_dict)

        # Bring all the relevant columns in the front.
        # cols = relevant_cols
        cols = relevant_cols + sorted(set(curr_df.columns) - set(relevant_cols))
        curr_df = curr_df[cols]

        dfs.append(curr_df)

    result_df = pd.concat(dfs)
    result_df = result_df.loc[result_df.groupby("config_id")[default_loss_name].idxmin().drop_duplicates(keep='last')]
    result_df = result_df.sort_values(by=default_loss_name)

    print("--------------------------------------------------------------------")
    print(f"result_df:\n{result_df[relevant_cols].drop(columns='config')}")
    # print(f"result_df:\n{result_df.drop(columns='config')}")

    # mem.log_memory(print, "after_save")
