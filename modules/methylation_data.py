import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory import Memory
from modules.api_handler import get_api_response
from modules.metadata import metadata_parquet_path

methylation_data_dir = "methylation_data"
# exmaple `gsm_id`: GSM1007129
gsm_id_pattern = "<gsm_id>"
base_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?view=data&acc={gsm_id_pattern}&db=GeoDb_blob87"


def append_to_df(df, gsm_id, cpg_sites_df):

    # # &&&
    gsm_id = "GSM1401026"

    url = base_url.replace(gsm_id_pattern, gsm_id)
    response = get_api_response(request_type="GET", url=url)

    pattern = r'<strong>ID_REF</strong><strong>\tVALUE</strong>(?:<strong>.*</strong>)?\n(.*?)(?=<br>)'
    match = re.search(pattern, response.text, re.DOTALL)

    if not match:
        raise ValueError(f"No matching data found for gsm_id: {gsm_id}")

    data_ = [line.strip().split('\t') for line in match.group(0).strip().splitlines()]
    data = data_[1:]

    if len(data[0]) == 2:
        curr_df = pd.DataFrame(data, columns=['cpg_site', gsm_id])
    elif len(data[0]) == 3:
        curr_df = pd.DataFrame(data, columns=['cpg_site', gsm_id, "detection_pval"])[['cpg_site', gsm_id]]
    else:
        raise ValueError(f"Bad number of columns for curr_df '{len(data[0])}'. gsm_id: '{gsm_id}'.")

    curr_df = curr_df[curr_df['cpg_site'].isin(cpg_sites_df.index)]
    curr_df[gsm_id] = curr_df[gsm_id].replace("NULL", -1.0).astype(float)
    curr_df = curr_df.set_index('cpg_site').sort_index()

    missing_sites = cpg_sites_df.index.difference(curr_df.index)
    if not missing_sites.empty:
        raise ValueError(f"The following CpG sites from cpg_sites_df are missing in curr_df: {missing_sites.tolist()}")

    if df.empty:
        df = curr_df
    else:
        if not df.index.equals(curr_df.index):
            raise ValueError("Mismatch in cpg_site values between DataFrames.")
        df = df.merge(curr_df, left_index=True, right_index=True)

    return df


def main(max_iters, save_every, override):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    mem.log_memory(print, "before____")
    dt_start = datetime.now()

    metadata_gsm_ids = pd.read_parquet(metadata_parquet_path).index.tolist()
    print(f"Got '{len(metadata_gsm_ids)}' metadata_gsm_ids.")

    methylation_data_parquet_path = f"resources/methylation_data.parquet"
    # methylation_data_parquet_path = f"resources/methylation_data_{max_iters}.parquet"
    path_ = Path(methylation_data_parquet_path)
    if path_.exists() and not override:
        df = pd.read_parquet(methylation_data_parquet_path)
        existing_gsm_ids = df.columns.tolist()
    else:
        df = pd.DataFrame()
        existing_gsm_ids = []

    gsm_ids = sorted(list(set(metadata_gsm_ids) - set(existing_gsm_ids)))
    # gsm_ids = [
    #     # "GSM1007129",
    #     # "GSM1272122",
    #     "GSM1272123",
    # ]
    # df = pd.DataFrame()
    print(f"Got '{len(gsm_ids)}' gsm_ids.")

    cpg_sites_path = f"resources/cpg_sites.parquet"
    cpg_sites_df = pd.read_parquet(cpg_sites_path).sort_index()  # sorting just in case.
    print(f"Got '{len(cpg_sites_df)}' cpg sites.")

    mem.log_memory(print, "before_fetching")

    dt_curr_start = datetime.now()
    for i, gsm_id in enumerate(gsm_ids):
        i += 1
        if i > max_iters:
            print(f"Reached {i} max iterations.")
            break
        df = append_to_df(df, gsm_id, cpg_sites_df)
        # sleep(1)
        if df.shape[1] % save_every == 0:
            print("----------------------------")
            print(f"df.shape: {df.shape}")
            df.to_parquet(methylation_data_parquet_path, engine='pyarrow', index=True)
            mem.log_memory(print, gsm_id)
            dt_curr_end = datetime.now()
            print(f"Diff   runtime: {dt_curr_end - dt_curr_start}")
            print(f"So far runtime: {dt_curr_end - dt_start}")
            dt_curr_start = datetime.now()

    mem.log_memory(print, "end")
    print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    # for max_iters in [50, 60, 70, 80, 90, 100]:
    main(10000, 10, override=False)
