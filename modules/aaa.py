"""
Apparently there are two ways to get beta values for studies that don't have data in series_matrix.
1. matrix_normalized (this file handles that).
2. IDAT files.
"""

import gzip
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.core.computation.expr import intersection

from memory import Memory
from modules.api_handler import get_api_response
from modules.gse_ids import gse_ids
from modules.metadata import metadata_parquet_path, cpg_site_id_str, gsm_id_str


def get_sample_title_to_gsm_id_mapping(path):

    samples_title_ls = []
    series_sample_id_ls = []
    sample_geo_accession_ls = []
    series_matrix_table_ls = []

    with open(path, "r") as file:
        for line in file:
            if line.startswith("!Sample_title"):
                samples_title_ls = [token.strip() for token in line.split("\t")]
                assert samples_title_ls[0] == "!Sample_title"
                pattern = r"(sample\d+)"
                samples_title_ls = [re.search(pattern, s).group(0) for s in samples_title_ls[1:]]
            elif line.startswith("!Series_sample_id"):
                series_sample_id_ls = [token.strip().replace('"', '') for token in line.split("\t")]
                series_sample_id_ls = [gsm_id.strip() for gsm_id in series_sample_id_ls[1].split()]
            elif line.startswith("!Sample_geo_accession"):
                sample_geo_accession_ls = [token.replace('"', '').strip() for token in line.split("\t")][1:]
            elif line.startswith('"ID_REF"'):
                series_matrix_table_ls = [token.replace('"', '').strip() for token in line.split("\t")][1:]
            else:
                continue

    assert sample_geo_accession_ls == series_matrix_table_ls == series_sample_id_ls
    assert len(sample_geo_accession_ls) == len(samples_title_ls)

    zip_ = zip(samples_title_ls, sample_geo_accession_ls)
    sample_title_to_gsm_id = dict(zip_)
    return sample_title_to_gsm_id


def main():
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

    # We are still using the series_matrix, even for studies where the
    #   actual methylation data is sitting in IDAT files, for the mapping between
    #   the gsm_ids and sample titles.
    temp_methylation_path = "resources/GSE125105_series_matrix.txt"
    sample_title_mapping = get_sample_title_to_gsm_id_mapping(temp_methylation_path)

    # temp_methylation_path = "resources/GSE125105_matrix_normalized.txt"
    # headers = None
    # lines_by_cpg_site_id = []
    # processed = 0
    # print_progress_every = 10000
    # with open(temp_methylation_path, "r") as file:
    #     headers = [header.strip() for header in file.readline().split()]
    #     for line in file:
    #         values = [val.strip() for val in line.split()][1:]  # remove the leading index.
    #         if values[0] in cpg_sites_df.index.tolist():
    #             lines_by_cpg_site_id.append(values)
    #         processed += 1
    #         if processed % print_progress_every == 0:
    #             print(f"Processed lines so far: {processed}")
    #     print(f"Total lines processed: {processed}")

    temp_methylation_path = "resources/temp_methylation.parquet"
    # df = pd.DataFrame(lines_by_cpg_site_id, columns=headers)
    # df.to_parquet(temp_methylation_path, engine='pyarrow', index=False)
    df = pd.read_parquet(temp_methylation_path)
    df = df.loc[:, ~df.columns.str.endswith('_DetectionPval')]
    df = df.rename(columns={"ID_REF": cpg_site_id_str}).set_index(cpg_site_id_str).sort_index()
    # matrix_normalized_df
    mat_norm_df = df.apply(pd.to_numeric, errors='coerce')

    missing_cpg_sites = sorted(set(cpg_sites_df.index.tolist()) - set(mat_norm_df.index.tolist()))
    if missing_cpg_sites:
        print(f"The following cpg_sites are missing from the normalized table: {missing_cpg_sites}")

    mat_norm_df.rename(columns=sample_title_mapping, inplace=True)
    mat_norm_df = mat_norm_df[mat_norm_df.columns.intersection(metadata_gsm_ids)]

    na_count = mat_norm_df.isna().sum().sum()
    if na_count > 0:
        print(f"Number of cells with NaN values: {na_count}")

    # --- this is where we start to process the IDAT stuff.

    ## Read in the sample_sheet_meta_data.parquet and get the mapping b/n sample_id -> gsm_id

    sample_sheet_meta_path = "resources/GSE125105_RAW_few/sample_sheet_meta_data.parquet"
    df_ = pd.read_parquet(sample_sheet_meta_path)
    df_ = df_.rename(columns={"GSM_ID": "gsm_id", "Sample_ID": "sample_id"})
    df_ = df_[["gsm_id", "sample_id"]]
    sample_id_to_gsm_id = df_.set_index('sample_id')['gsm_id'].to_dict()

    # &&& we need to re-enable this when we get the full gsm_id set.
    # df_ = df_[df_.columns.intersection(metadata_gsm_ids)]

    ## Read in the beta values from beta_values.parquet

    # these are run through the methylprep pipline with `run_pipeline`.
    beta_values_idat_path = "resources/GSE125105_RAW_few/beta_values.parquet"
    df_ = pd.read_parquet(beta_values_idat_path)
    df_ = df_[df_.index.isin(cpg_sites_df.index)].sort_index()
    idat_df = df_.rename(columns=sample_id_to_gsm_id)

    mat_norm_filt_df = mat_norm_df[mat_norm_df.columns.intersection(idat_df.columns)]

    # rename columns
    mat_norm_filt_df = mat_norm_filt_df.add_suffix("_mat_norm")
    idat_df = idat_df.add_suffix("_idat")

    df = pd.merge(idat_df, mat_norm_filt_df, left_index=True, right_index=True, how='left')
    df = df[sorted(df.columns)]





    fjdkfjdk = 1














    lines = [line.strip() for line in lines]
    lines = [line.split("\t") for line in lines]

    fdjkfd = 1





    mem.log_memory(print, "end")
    print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    main()
