from pathlib import Path
import json
import numpy as np
import pandas as pd
from io import StringIO
from memory import Memory
from modules.ml_pipeline import default_loss_name

if __name__ == '__main__':

    pd.set_option('display.max_columns', 10)
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
    # path = Path(f"result_artifacts/result_df.parquet")
    path = Path(f"result_artifacts_temp/result_df.parquet")

    df = pd.read_parquet(path)
    # df = pd.read_pickle(path)

    # ## get sample_id_to_gsm_id
    # sample_id_to_gsm_id = df.set_index("Sample_ID")["GSM_ID"].to_dict()

    # # Find out the best config so far.
    # df = df.sort_values(by=default_loss_name)
    # ser = df.iloc[0]
    # loss = ser[default_loss_name]
    # config = json.loads(ser["config"])

    fjdkfjdk = 1

    df = df.round(3).astype(np.float32)

    # path = "resources/methylation_data_2.parquet"
    path = "resources_methylprep/GSE102177_process_pandas_1.3.5/beta_values.pkl"

    df2 = pd.read_pickle(path)

    df2 = df2.round(3).astype(np.float32)

    df = df.fillna(-9999)
    df2 = df2.fillna(-9999)

    # Compare dfs element-wise
    comparison = df.ne(df2)  # Compare DataFrames element-wise
    mismatch_indices = np.where(comparison)  # Get the indices of mismatches
    if mismatch_indices[0].size > 0:
        for i in range(len(mismatch_indices[0])):
            row_index = mismatch_indices[0][i]  # First mismatched row
            col_index = mismatch_indices[1][i]  # First mismatched column

            # Get row and column labels
            row_label = df.row_index[row_index]
            col_label = df.columns[col_index]

            # Get the differing values
            value1 = df.iloc[row_index, col_index]
            value2 = df2.iloc[row_index, col_index]

            print(f"Found diff '{i}' at row '{row_label}', column '{col_label}':")
            print(f"Value in df : {value1:,.3f}")
            print(f"Value in df2: {value2:,.3f}")
    else:
        print("No differences found.")

    pd.testing.assert_frame_equal(df, df2, check_exact=False)

    fdjkfd = 1

    # # sort by missing values per gsm
    # row_nan_counts = df.isna().sum(axis=1)
    # row_nan_counts.name = "nan_count"
    # row_nan_counts = row_nan_counts.sort_values(ascending=False)
    #
    # # sort by missing values per cpg_site
    # col_nan_counts = df.isna().sum(axis=0)
    # col_nan_counts.name = "nan_count"
    # col_nan_counts = col_nan_counts.sort_values(ascending=False)

    print(df)

    mem.log_memory(print, "after_save")



















# &&&
