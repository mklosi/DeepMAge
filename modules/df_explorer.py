import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from io import StringIO
from memory import Memory
from modules.ml_common import get_config_id
from modules.ml_optuna_1 import config_id_str
from modules.ml_pipeline import default_loss_name


if __name__ == '__main__':

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.min_rows', 20)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    mem.log_memory(print, "before____")

    # path = "resources/metadata.parquet"
    # path = "resources/metadata_derived.parquet"
    # path = "resources/methylation_data.parquet"
    # path = "resources/GSE125105_RAW_few/beta_values.parquet"
    # path = "resources/GSE125105_RAW_few/control_probes.parquet"
    # path = "resources/GSE125105_RAW_few/m_values.parquet"
    # path = "resources/GSE125105_RAW_few/noob_meth_values.parquet"
    # path = "resources/GSE125105_RAW_few/noob_unmeth_values.parquet"
    # path = "resources/GSE125105_RAW_few/sample_sheet_meta_data.parquet"
    # path = "resources_methylprep/GSE102177_download_pandas_1.3.5/GPL13534/beta_values.pkl"
    # path = "resources_methylprep/GSE102177_download_pandas_1.3.5/GPL13534/GSE102177_GPL13534_meta_data.pkl"
    # path = Path(f"result_artifacts/result_df_study-1.parquet")
    path = Path(f"result_artifacts/result_df.parquet")
    # path = Path(f"result_artifacts_temp/result_df.parquet")

    df = pd.read_parquet(path)
    # df = pd.read_pickle(path)

    # check for duplications
    df_deduped = df.loc[df.groupby(["config_id"])['datetime_start'].idxmin()]

    fdjkfjddf = 1

    # Find out the best config so far.
    df = df.sort_values(by=default_loss_name)
    ser = df.iloc[0]
    loss = ser[default_loss_name]
    config_json = ser["config"]
    config = json.loads(config_json)
    config_json_indent = json.dumps(config, indent=4)
    print(f"Best config so far:\n{config_json_indent}\n{default_loss_name}: {loss}")

    # Find out config and loss for a specific config_id.
    config_id = "1178f53710f87a766f1d9366362aec99"  # param
    if config_id in set(df[config_id_str].to_list()):
        df_ = df[df[config_id_str] == config_id]
        assert len(df_) == 1
        ser = df_.iloc[0]
        loss = ser[default_loss_name]
        config_json = ser["config"]
        config = json.loads(config_json)
        config_json_indent = json.dumps(config, indent=4)
        print(f"Config for config_id '{config_id}':\n{config_json_indent}\n{default_loss_name}: {loss}")
    else:
        print(f"Config for config_id '{config_id}' not found.")

    fjdkfjdk = 1

    # ## Compare dfs.
    #
    # df = df.round(3).astype(np.float32)
    #
    # # path = "resources/methylation_data_2.parquet"
    # path = "resources_methylprep/GSE102177_process_pandas_1.3.5/beta_values.pkl"
    #
    # df2 = pd.read_pickle(path)
    #
    # df2 = df2.round(3).astype(np.float32)
    #
    # df = df.fillna(-9999)
    # df2 = df2.fillna(-9999)
    #
    # # Compare dfs element-wise
    # comparison = df.ne(df2)  # Compare DataFrames element-wise
    # mismatch_indices = np.where(comparison)  # Get the indices of mismatches
    # if mismatch_indices[0].size > 0:
    #     for i in range(len(mismatch_indices[0])):
    #         row_index = mismatch_indices[0][i]  # First mismatched row
    #         col_index = mismatch_indices[1][i]  # First mismatched column
    #
    #         # Get row and column labels
    #         row_label = df.row_index[row_index]
    #         col_label = df.columns[col_index]
    #
    #         # Get the differing values
    #         value1 = df.iloc[row_index, col_index]
    #         value2 = df2.iloc[row_index, col_index]
    #
    #         print(f"Found diff '{i}' at row '{row_label}', column '{col_label}':")
    #         print(f"Value in df : {value1:,.3f}")
    #         print(f"Value in df2: {value2:,.3f}")
    # else:
    #     print("No differences found.")
    #
    # # Compare dfs holistically.
    # pd.testing.assert_frame_equal(df, df2, check_exact=False)

    mem.log_memory(print, "after_save")
