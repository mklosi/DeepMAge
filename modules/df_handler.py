import pandas as pd
from io import StringIO
from memory import Memory

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    mem.log_memory(print, "before____")

    path = "resources/metadata.parquet"
    # path = "resources/methylation_data.parquet"
    df = pd.read_parquet(path)

    # path = "resources/GSM1272122-8224.txt"
    # df = pd.read_csv(path, sep='\t', comment='#', header=0, names=['cpg_site', 'GSM1272122', '1001_detection_pval'])

    # path = "resources/TableS2_DeepMAge_features_w_importance_scores.tsv"
    # df = pd.read_csv(
    #     path, sep='\t', comment='#', header=0, names=['cpg_site', 'importance'], dtype={'Importance': float}
    # ).set_index('cpg_site').sort_index()

    print(df)

    mem.log_memory(print, "after_save")
