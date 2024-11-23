"""
This code will get methylation amounts per study using the <>_series_matrix.txt.
More efficient, but some studies have their data in IDAT format, not in _series_matrix.txt files.
"""

import gzip
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory import Memory
from modules.api_handler import get_api_response
from modules.gse_ids import gse_ids  # &&& is this really necessary?
from modules.metadata import metadata_parquet_path, cpg_site_id_str, gsm_id_str

gse_id_three_digits_pattern = "<three_digits>"
gse_id_pattern = "<gse_id>"
# this is the series_matrix url.
series_matrix_source = {
    "source_type": "series_matrix",
    "source_url": (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{gse_id_three_digits_pattern}nnn/{gse_id_pattern}/matrix/"
        f"{gse_id_pattern}_series_matrix.txt.gz"
    )
}
matrix_normalized_source = {
    "source_type": "matrix_normalized",  # methyl data is split by tabs
    "source_url": (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{gse_id_three_digits_pattern}nnn/{gse_id_pattern}/suppl/"
        f"{gse_id_pattern}_matrix_normalized.txt.gz"
    )
}
matrix_processed_source = {
    "source_type": "matrix_processed",  # methyl data is split by commas
    "source_url": (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{gse_id_three_digits_pattern}nnn/{gse_id_pattern}/suppl/"
        f"{gse_id_pattern}_MatrixProcessed.csv.gz"
    )
}
gse_id_to_extract_source = { # &&& turn into comprehension. if.
    "GSE125105": matrix_normalized_source,
    "GSE128235": matrix_normalized_source,
    "GSE59065": matrix_processed_source,
    "GSE61496": matrix_normalized_source,
    "GSE77696": matrix_normalized_source,
}

mem = Memory(noop=False)


def get_sample_title_to_gsm_id_mapping(file):

    samples_title_ls = []
    series_sample_id_ls = []
    sample_geo_accession_ls = []
    series_matrix_table_ls = []

    for line in file:
        if line.startswith("!Sample_title"):
            samples_title_ls = [token.strip() for token in line.split("\t")]
            assert samples_title_ls[0] == "!Sample_title"
            pattern = r"sample ?(\d+)"
            samples_title_ls = [
                f"sample{re.search(pattern, s).group(1)}"
                for s in samples_title_ls[1:]
            ]
        elif line.startswith("!Series_sample_id"):
            series_sample_id_ls = [token.strip().replace('"', '') for token in line.split("\t")]
            series_sample_id_ls = [gsm_id.strip() for gsm_id in series_sample_id_ls[1].split()]
        elif line.startswith("!Sample_geo_accession"):
            sample_geo_accession_ls = [token.replace('"', '').strip() for token in line.split("\t")][1:]
        elif line.startswith('"ID_REF"'):
            series_matrix_table_ls = [token.replace('"', '').strip() for token in line.split("\t")][1:]
            break
        else:
            continue

    assert sample_geo_accession_ls == series_matrix_table_ls == series_sample_id_ls
    assert len(sample_geo_accession_ls) == len(samples_title_ls)

    zip_ = zip(samples_title_ls, sample_geo_accession_ls)
    sample_title_to_gsm_id = dict(zip_)
    return sample_title_to_gsm_id


def get_series_matrix_df(file, cond_):
    extracting = False
    header = True
    extracted_lines = []

    for line in file:
        if line.startswith("!series_matrix_table_begin"):
            extracting = True
            continue  # Skip the marker line itself
        elif line.startswith("!series_matrix_table_end"):
            break  # Stop reading after the end marker

        if extracting:
            ls_ = line.replace('"', '').split('\t')
            if header or ls_[0] in cond_:
                # noinspection PyTypeChecker
                extracted_lines.append(ls_)
                header = False

    df = pd.DataFrame(extracted_lines[1:], columns=extracted_lines[0])

    return df


def get_matrix_normalized_df(file, cond_):
    lines_by_cpg_site_id = []
    processed = 0
    print_progress_every = 10000
    columns = [header.strip() for header in file.readline().split()]
    for line in file:
        values = [val.strip() for val in line.split()][1:]  # remove the leading index.
        if values[0] in cond_: # &&& filter.
        # if True:
            lines_by_cpg_site_id.append(values)
        processed += 1
        if processed % print_progress_every == 0:
            print(f"Processed lines so far: {processed}")
    print(f"Total lines processed: {processed}")

    df = pd.DataFrame(lines_by_cpg_site_id, columns=columns)
    df = df.loc[:, ~df.columns.str.endswith('_DetectionPval')]

    return df


def get_matrix_processed_df(file, cond_):
    lines_by_cpg_site_id = []
    processed = 0
    print_progress_every = 10000
    columns = [header.strip().replace('"', '') for header in file.readline().split(",")] # &&& diff x2
    for line in file:
        values = [val.strip().replace('"', '') for val in line.split(",")]  # remove the leading index.  # &&& diff x3
        if values[0] in cond_: # &&& filter.
        # if True:
            lines_by_cpg_site_id.append(values)
        processed += 1
        if processed % print_progress_every == 0:
            print(f"Processed lines so far: {processed}")
    print(f"Total lines processed: {processed}")

    df = pd.DataFrame(lines_by_cpg_site_id, columns=columns)
    df = df.loc[:, ~df.columns.str.endswith('_DetectionPval')]

    return df


def append_to_df(df, gse_id, gsm_ids, cpg_sites_df):

    # &&& change to cond_ outside of this.
    cond_ = set(cpg_sites_df.index.tolist())
    column_renames = {'ID_REF': cpg_site_id_str}

    three_digits = gse_id[:-3]
    series_matrix_url = (
        series_matrix_source["source_url"]
        .replace(gse_id_three_digits_pattern, three_digits)
        .replace(gse_id_pattern, gse_id)
    )

    # dt_response = datetime.now()
    response = get_api_response(request_type="GET", url=series_matrix_url)
    content = io.BytesIO(response.content)
    # mem.log_memory(print, "series_matrix")
    # print(f"series_matrix response runtime: {datetime.now() - dt_response}") # &&& remove these. they are just distracting.

    with gzip.open(content, 'rt') as file:
        curr_df = get_series_matrix_df(file, cond_)
        if curr_df.empty:
            if gse_id not in gse_id_to_extract_source:
                raise ValueError(f"For gse_id '{gse_id}', both curr_df is empty and not IDAT.")

    if gse_id in gse_id_to_extract_source:
        content = io.BytesIO(response.content)
        with gzip.open(content, 'rt') as file:
            # &&&
            # sample_title_mapping = get_sample_title_to_gsm_id_mapping(file)
            sample_title_mapping = {}
            column_renames.update(sample_title_mapping)

        mat_norm_url = (
            gse_id_to_extract_source[gse_id]["source_url"]
            .replace(gse_id_three_digits_pattern, three_digits)
            .replace(gse_id_pattern, gse_id)
        )

        # dt_response = datetime.now()
        response = get_api_response(request_type="GET", url=mat_norm_url)
        content = io.BytesIO(response.content)
        # mem.log_memory(print, "matrix_normalized")
        # print(f"matrix_normalized response runtime: {datetime.now() - dt_response}")

        # content = 'resources/GSE125105_matrix_normalized_small.txt.gz' # &&&

        with gzip.open(content, 'rt') as file:
            if gse_id_to_extract_source[gse_id]["source_type"] == "matrix_normalized":
                curr_df = get_matrix_normalized_df(file, cond_)
            elif gse_id_to_extract_source[gse_id]["source_type"] == "matrix_processed":
                curr_df = get_matrix_processed_df(file, cond_)
            else:
                raise ValueError(f"Bad source_type: {gse_id_to_extract_source[gse_id]['source_type']}")

    curr_df.columns = curr_df.columns.str.strip()
    curr_df = curr_df.rename(columns=column_renames)
    curr_df = curr_df.set_index(cpg_site_id_str).sort_index()
    curr_df = curr_df[sorted(curr_df.columns)]

    missing_sites = cpg_sites_df.index.difference(curr_df.index)
    if not missing_sites.empty:
        ls_ = missing_sites.tolist()
        print(
            f"The following '{len(ls_)}' CpG sites from cpg_sites_df are missing in curr_df '{ls_}'. "
            f"Appending NaN values for each sample."
        )

        # &&&
        if len(ls_) == 1000:
            raise ValueError(
                f"The following '{len(ls_)}' CpG sites from cpg_sites_df are missing in curr_df '{ls_}'. "
                f"Appending NaN values for each sample."
            )

        curr_df = curr_df.reindex(cpg_sites_df.index).sort_index()

    curr_df = curr_df.apply(pd.to_numeric, errors='coerce')
    curr_df = curr_df.T
    curr_df.index.name = gsm_id_str
    curr_df.columns.name = "cpg_site"  # &&& this needs to change and go somewhere else.

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

    # &&&
    gse_ids = [
        # "GSE102177",
        # "GSE103911",
        # "GSE105123",
        # "GSE106648",
        # "GSE107459",
        # "GSE107737",
        # "GSE112696",
        # "GSE19711",
        # "GSE20067",
        # "GSE27044",
        # "GSE30870",
        # "GSE34639",
        # "GSE37008",
        # "GSE40279",
        # "GSE41037",
        # "GSE52588",
        # "GSE53740",
        # "GSE58119",
        # "GSE67530",
        # "GSE77445",
        # "GSE79329",
        # "GSE81961",
        # "GSE84624",
        # "GSE87582",
        # "GSE87640",
        # "GSE97362",
        # "GSE98876",
        # "GSE99624",

        # "GSE125105",
        # "GSE128235",
        "GSE59065",
        # "GSE61496",
        # "GSE77696",
    ]
    unseen_gse_ids = sorted(set(gse_ids) - set(seen_gse_ids))
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
    main(override=True)  # &&&
