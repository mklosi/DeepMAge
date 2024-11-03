import gzip
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory import Memory
from modules.api_handler import get_api_response
from modules.gse_ids import gse_ids
from modules.metadata import metadata_parquet_path, cpg_site_id_str, gsm_id_str

gse_id_three_digits_pattern = "<three_digits>"
gse_id_pattern = "<gse_id>"
base_url = (
    f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_id_three_digits_pattern}nnn/{gse_id_pattern}/matrix/"
    f"{gse_id_pattern}_series_matrix.txt.gz"
)


def extract_matrix_table(lines):
    extracting = False
    extracted_lines = []

    for line in lines:
        if line.startswith("!series_matrix_table_begin"):
            extracting = True
            continue  # Skip the marker line itself
        elif line.startswith("!series_matrix_table_end"):
            break  # Stop reading after the end marker

        if extracting:
            extracted_lines.append(line)

    return extracted_lines


def append_to_df(df, gse_id, gsm_ids, cpg_sites_df):

    three_digits = gse_id[:-3]
    url = base_url.replace(gse_id_three_digits_pattern, three_digits).replace(gse_id_pattern, gse_id)

    dt_start = datetime.now()
    response = get_api_response(request_type="GET", url=url)
    print(f"response runtime: {datetime.now() - dt_start}")

    with gzip.open(io.BytesIO(response.content), 'rt') as file:

        dt_start = datetime.now()
        lines = file.readlines()  # This will give you a list of strings, one per line
        print(f"readlines runtime: {datetime.now() - dt_start}")

        dt_start = datetime.now()
        matrix_table_lines = extract_matrix_table(lines)
        print(f"extract_matrix_table runtime: {datetime.now() - dt_start}")

        dt_start = datetime.now()
        data = [line.replace('"', '').split('\t') for line in matrix_table_lines]
        # data = [[element.strip() for element in line.replace('"', '').split('\t')] for line in matrix_table_lines]
        print(f"data runtime: {datetime.now() - dt_start}")

        columns = data[0]
        data = data[1:]

        dt_start = datetime.now()
        cond_ = set(cpg_sites_df.index.tolist())
        data = [ls_ for ls_ in data if ls_[0] in cond_]
        print(f"filter runtime: {datetime.now() - dt_start}")

        dt_start = datetime.now()
        curr_df = pd.DataFrame(data, columns=columns)
        print(f"pd.DataFrame runtime: {datetime.now() - dt_start}")

        dt_start = datetime.now()
        curr_df = curr_df.rename(columns={'ID_REF': cpg_site_id_str})
        print(f"rename_col runtime: {datetime.now() - dt_start}")

        # dt_start = datetime.now()
        # curr_df = curr_df[curr_df[cpg_site_id_str].isin(cpg_sites_df.index)]
        # print(f"filter2 runtime: {datetime.now() - dt_start}")

        curr_df.columns = curr_df.columns.str.strip()

        curr_df = curr_df.set_index(cpg_site_id_str).sort_index()

        missing_sites = cpg_sites_df.index.difference(curr_df.index)
        if not missing_sites.empty:
            print(
                f"The following CpG sites from cpg_sites_df are missing in curr_df '{missing_sites.tolist()}'. "
                f"Appending NaN values for each sample."
            )
            curr_df = curr_df.reindex(cpg_sites_df.index)
            curr_df.sort_index()

    # curr_df = curr_df.astype(float)
    curr_df = curr_df.apply(pd.to_numeric, errors='coerce')
    curr_df = curr_df.T
    curr_df.index.name = gsm_id_str

    curr_df = curr_df[curr_df.index.isin(gsm_ids)]
    print(f"Retrieved values for '{len(curr_df)}' new gsm_ids.")

    if df.empty:
        df = curr_df
    else:
        if not df.columns.equals(curr_df.columns):
            raise ValueError("Mismatch in cpg_site values between DataFrames.")
        df = pd.concat([df, curr_df])
        df_duplicates = df[df.index.duplicated(keep=False)]
        if len(df_duplicates) > 0:
            raise ValueError(f"Found '{len(df_duplicates)}' duplicate rows when concatenating the dfs.")
        df = df.sort_index()

    return df


def main(override):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 200)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    mem.log_memory(print, "before____")
    dt_start = datetime.now()

    metadata_gsm_ids = pd.read_parquet(metadata_parquet_path).index.tolist()
    print(f"Got '{len(metadata_gsm_ids)}' metadata_gsm_ids.")

    cpg_sites_path = f"resources/cpg_sites.parquet"
    cpg_sites_df = pd.read_parquet(cpg_sites_path).sort_index()  # sorting just in case.
    print(f"Got '{len(cpg_sites_df)}' cpg sites.")

    mem.log_memory(print, "before_fetching")

    methylation_data_parquet_path = Path("resources/methylation_data.parquet")
    # methylation_data_parquet_path = Path("resources/methylation_data_{max_iters}.parquet")
    seen_gse_id_path = Path("states/seen_gse_id.txt")
    if methylation_data_parquet_path.exists() and seen_gse_id_path.exists() and not override:
        # rows: gsm_ids. cols: cpg_site_ids (1000 count).
        df = pd.read_parquet(methylation_data_parquet_path)
        seen_gse_ids = seen_gse_id_path.read_text().splitlines()
    else:
        methylation_data_parquet_path.unlink(missing_ok=True)
        seen_gse_id_path.unlink(missing_ok=True)
        df = pd.DataFrame()
        seen_gse_ids = []

    unseen_gse_ids = sorted(set(gse_ids) - set(seen_gse_ids))
    unseen_gse_ids = [
        "GSE106648",
    ]
    print(f"Got '{len(unseen_gse_ids)}' unseen_gse_ids.")

    dt_curr_start = datetime.now()
    for gse_id in unseen_gse_ids:
        print("----------------------------")
        print(f"Running gse_id: {gse_id}")
        df = append_to_df(df, gse_id, metadata_gsm_ids, cpg_sites_df)
        print(f"df.shape: {df.shape}")
        df.to_parquet(methylation_data_parquet_path, engine='pyarrow', index=True)
        seen_gse_ids.append(gse_id)
        seen_gse_id_path.write_text("\n".join(seen_gse_ids) + "\n")
        mem.log_memory(print, gse_id)
        dt_curr_end = datetime.now()
        print(f"Diff   runtime: {dt_curr_end - dt_curr_start}")
        print(f"So far runtime: {dt_curr_end - dt_start}")
        dt_curr_start = datetime.now()

    mem.log_memory(print, "end")
    print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    main(override=True) # &&&
