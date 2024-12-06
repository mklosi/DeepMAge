import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from io import StringIO
from memory import Memory
from modules.ml_common import get_config_id
from modules.ml_optuna_1 import study_db_url
from modules.ml_pipeline import default_loss_name

if __name__ == '__main__':

    pd.set_option('display.max_columns', 10)
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

    study_summaries = optuna.study.get_all_study_summaries(storage=study_db_url)
    print(f"Study count: {len(study_summaries)}")
    for study_summary in study_summaries:
        print(f"\tstudy_name: {study_summary.study_name}")
        study = optuna.load_study(study_name=study_summary.study_name, storage=study_db_url)

        curr_df = study.trials_dataframe()

        # Remove prefix 'user_attrs_' from any columns that have it.
        cols = {col: col.replace('user_attrs_', '') for col in curr_df.columns}
        curr_df = curr_df.rename(columns=cols)

        # Bring all the relevant columns in the front.
        relevant_cols = ["config_id", "datetime_start", "duration", "mse", "mae", "medae", "config"]
        # cols = relevant_cols + sorted(set(result_df.columns) - set(relevant_cols))
        curr_df = curr_df[relevant_cols]

        dfs.append(curr_df)

    result_df = pd.concat(dfs)
    result_df = result_df.loc[result_df.groupby("config_id")[default_loss_name].idxmin().drop_duplicates(keep='last')]
    # result_df = result_df.set_index("config_id")
    result_df = result_df.sort_values(by=default_loss_name)

    print(f"result_df:\n{result_df.drop(columns='config')}")




    # print(f"fjdkfdjk: {len(}")


    fjdkfjdk = 1



    # mem.log_memory(print, "after_save")



















# &&&
