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

    # path = "resources/metadata.parquet"
    path = "resources/methylation_data_1.parquet"

    # path = "resources/GSE125105_RAW_few/beta_values.parquet"
    # path = "resources/GSE125105_RAW_few/control_probes.parquet"
    # path = "resources/GSE125105_RAW_few/m_values.parquet"
    # path = "resources/GSE125105_RAW_few/noob_meth_values.parquet"
    # path = "resources/GSE125105_RAW_few/noob_unmeth_values.parquet"
    # path = "resources/GSE125105_RAW_few/sample_sheet_meta_data.parquet"

    df = pd.read_parquet(path)

    # path = "resources/GSM1272122-8224.txt"
    # df = pd.read_csv(path, sep='\t', comment='#', header=0, names=['cpg_site', 'GSM1272122', '1001_detection_pval'])

    # path = "resources/TableS2_DeepMAge_features_w_importance_scores.tsv"
    # df = pd.read_csv(
    #     path, sep='\t', comment='#', header=0, names=['cpg_site', 'importance'], dtype={'Importance': float}
    # ).set_index('cpg_site').sort_index()

    # missing_se = pd.Series({index: row[row.isna()].index.tolist() for index, row in df.iterrows() if row.isna().any()})
    # missing_df = missing_se.to_frame(name="missing_columns")
    # missing_df["missing_count"] = missing_df["missing_columns"].apply(len)
    # missing_df = missing_df.sort_values(by="missing_count", ascending=False)

    path = "resources/methylation_data_2.parquet"
    df2 = pd.read_parquet(path)


    # df2.loc["GSM1401026", "cg00047050"] = 0.00882101

    pd.testing.assert_frame_equal(df, df2)






    # print(df)

    mem.log_memory(print, "after_save")
