import gzip
import io
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory import Memory
from modules.api_handler import get_api_response
from modules.gse_ids import gse_ids
from modules.metadata import metadata_parquet_path, cpg_site_id_str, gsm_id_str


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 200)
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    mem.log_memory(print, "before____")
    dt_start = datetime.now()








    mem.log_memory(print, "end")
    print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    main()
