import numpy as np
import pandas as pd
from io import StringIO
from memory import Memory
from modules.gse_ids_by_source import gse_id_to_source
from modules.metadata import metadata_parquet_path

if __name__ == '__main__':

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.min_rows', 20)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    mem = Memory(noop=False)
    # mem.log_memory(print, "before____")

    # metadata_gsm_ids = pd.read_parquet(metadata_parquet_path).index.tolist()
    # print(f"Got '{len(metadata_gsm_ids)}' metadata_gsm_ids.")
    #
    # cpg_sites_path = f"resources/cpg_sites.parquet"
    # cpg_sites_df = pd.read_parquet(cpg_sites_path).sort_index()  # sorting just in case.
    # print(f"Got '{len(cpg_sites_df)}' cpg sites.")

    metadata_df = pd.read_parquet("/Users/roise0r/intellij-projects/DeepMAge/resources/metadata.parquet")

    df = metadata_df[[
        "2.health_status",
        "3.sample",
        "3.study",
        # "jkjk"
        # "1.real_value",
        "3.actual_age_years",  # *
        # "2.age",  # *
        # "3.age",  # *
        # "4.real_age",
        # "1.pred_value",
        # "3.pred_value",  # *
        "3.predicted_age_years",  # *
    ]].copy()

    new_col = "type"
    df[new_col] = np.where(
        df["3.sample"].isin(["train", "ill_train"]),
        "train",
        "verification"
    )

    # Place new column in 3rd place
    cols = list(df.columns)
    cols.insert(2, cols.pop(cols.index(new_col)))
    df = df[cols]

    df["source"] = df["3.study"].map(gse_id_to_source).fillna("Unknown")

    col_rename = {
        "3.study": "gse_id",
        "source": "source",
        "type": "type",
        "3.actual_age_years": "actual_age_years",
        "3.predicted_age_years": "predicted_age_years",
    }
    df = df.drop(columns=list(set(df.columns) - set(col_rename.keys())))
    df = df.rename(columns=col_rename)
    df = df[col_rename.values()].sort_index()

    age_df = df

    # age_df.to_parquet("/Users/roise0r/intellij-projects/DeepMAge/resources/metadata_derived.parquet", engine='pyarrow', index=True)

    # # Compare Series element-wise
    # s1 = df["3.pred_value"].round(3)
    # s2 = df["3.predicted_age_years"].round(3)
    # comparison = s1.ne(s2)
    # mismatch_indices = np.where(comparison)
    # if mismatch_indices[0].size > 0:
    #     for i, index in enumerate(mismatch_indices[0]):
    #         row_label = s1.index[index]
    #
    #         value1 = s1.iloc[index]
    #         value2 = s2.iloc[index]
    #
    #         print(f"Found diff '{i}' at index '{row_label}':")
    #         print(f"Value in s1: {value1:,.3f}")
    #         print(f"Value in s2: {value2:,.3f}")
    # else:
    #     print("No differences found.")

    df = pd.read_parquet("/Users/roise0r/intellij-projects/DeepMAge/resources/methylation_data.parquet")

    df = df[df.index.isin(age_df[age_df["gse_id"].isin([
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
        # # "GSE34639",  #*
        # # "GSE37008",  #*
        # "GSE40279",
        # "GSE41037",
        # "GSE52588",
        # "GSE53740",
        # "GSE58119",
        "GSE67530",  # a very slight negative value.
        # "GSE77445",
        # "GSE79329",
        # # "GSE81961",  #*
        # "GSE84624",
        # "GSE87582",
        # "GSE87640",
        # "GSE97362",
        # "GSE98876",
        # "GSE99624",
        # "GSE125105",
        # "GSE128235",
        # "GSE59065",
        # "GSE61496",
        # "GSE77696",
    ])].index)]

    negative_cells = [
        (row_label, col_label, df.at[row_label, col_label])
        for row_label, col_label in zip(*np.where(df < 0))
    ]






    fdjkfdkfdj = 1






















# &&&
