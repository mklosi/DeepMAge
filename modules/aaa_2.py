"""
Apparently there are two ways to get beta values for studies that don't have data in series_matrix.
1. matrix_normalized.
2. IDAT files (this file handles that).
"""

import gzip
import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory import Memory
from modules.api_handler import get_api_response
from modules.gse_ids import gse_ids
from modules.metadata import metadata_parquet_path, cpg_site_id_str, gsm_id_str
from methylprep import run_pipeline


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

    # --- IDAT reading and structuring

    # # idat_directory = 'resources/GSE125105_RAW/'
    # idat_directory = 'resources/GSE125105_RAW_few/'
    #
    # run_pipeline(
    #     idat_directory,
    #     betas=True,   # Calculates beta values
    #     m_value=True,  # Calculates M-values
    #     make_sample_sheet=True,  # If you don't have a sample sheet
    #     file_format="parquet",
    # )

    # -------------------------------------













    fjdkfjdk = 1














    # lines = [line.strip() for line in lines]
    # lines = [line.split("\t") for line in lines]
    #
    # fdjkfd = 1
    #
    #
    #
    #
    #
    # mem.log_memory(print, "end")
    # print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    main()
