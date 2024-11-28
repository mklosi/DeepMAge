import numpy as np
import pandas as pd
from io import StringIO

from sklearn.preprocessing import MinMaxScaler

from memory import Memory
from modules.gse_ids_by_source import gse_id_to_source
from modules.metadata import metadata_parquet_path

if __name__ == '__main__':

    pd.set_option('display.max_columns', 6)
    pd.set_option('display.min_rows', 20)
    pd.set_option('display.max_rows', 20)
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

    methyl_df = pd.read_parquet("/Users/roise0r/intellij-projects/DeepMAge/resources/methylation_data.parquet")

    # df = df.round(2)

    gse_ids = [
        "GSE102177",
        "GSE103911",
        "GSE105123",
        "GSE106648",
        "GSE107459",
        "GSE107737",
        "GSE112696",
        "GSE19711",
        "GSE20067",
        "GSE27044",
        "GSE30870",
        "GSE34639",  #*
        "GSE37008",  #*
        "GSE40279",
        "GSE41037",
        "GSE52588",
        "GSE53740",
        "GSE58119",
        "GSE67530",  # a very slight negative value. [('GSM1648898', 'cg11041457', -0.01), ('GSM1648973', 'cg11041457', -0.01)]
        "GSE77445",
        "GSE79329",
        "GSE81961",  #*
        "GSE84624",
        "GSE87582",
        "GSE87640",
        "GSE97362",
        "GSE98876",
        "GSE99624",
        "GSE125105",
        "GSE128235",
        "GSE59065",
        "GSE61496",
        "GSE77696",
    ]

    # for gse_id in gse_ids:

    df = methyl_df[methyl_df.index.isin(age_df[age_df["gse_id"].isin(gse_ids)].index)].copy()

    # ## Add nan sample, so I can test the filtering later.
    # new_gsm_id = 'GSM0'
    # nan_row = pd.DataFrame([[np.nan] * df.shape[1]], columns=df.columns, index=[new_gsm_id])
    # df = pd.concat([df, nan_row])

    # ## Add nan cpg_site.
    # df["new_cpg_site"] = np.nan

    na_cells = [
        (df.index[row_index], df.columns[col_index], df.iat[row_index, col_index])
        for row_index, col_index in zip(*np.where(df.isna()))
    ]

    negative_cells = [
        (df.index[row_index], df.columns[col_index], df.iat[row_index, col_index])
        for row_index, col_index in zip(*np.where(df < 0))
    ]

    # manual_norm_df = (df - df.min().min()) / (df.max().max() - df.min().min())

    # scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # norm_df = pd.DataFrame(
    #     scaler.fit_transform(df),
    #     columns=df.columns,
    #     index=df.index  # Preserve the original index
    # )
    #
    # min_max_df = pd.DataFrame(
    #     {col: [df[col].min(), df[col].max()] for col in df.columns},
    #     index=["min", "max"]
    # )
    # min_max_norm_df = pd.DataFrame(
    #     {col: [norm_df[col].min(), norm_df[col].max()] for col in norm_df.columns},
    #     index=["min", "max"]
    # )

    ## Show which gsm_ids have missing values and the corresponding cpg_sites that contain those nans.
    nan_counts = df.isna().sum(axis=1)  # Count NaNs per GSM ID
    gsm_ids_with_nans = nan_counts[nan_counts > 0]
    nan_cpg_sites = df.apply(lambda row: ','.join(row.index[row.isna()]), axis=1)
    gsm_ids_with_nans_df = pd.DataFrame({
        'gsm_id': gsm_ids_with_nans.index,
        'nan_count': gsm_ids_with_nans.values,
        'cpg_sites_with_nan': nan_cpg_sites[gsm_ids_with_nans.index].values
    })

    gsm_ids_with_all_nans = df[df.isna().all(axis=1)]
    if gsm_ids_with_all_nans.shape[0] != 0:
        print(f"There are '{gsm_ids_with_all_nans.shape[0]}' gsm_ids with all nans for '{len(gse_ids)}' gse_ids.")
        # print(f"There are '{gsm_ids_with_all_nans.shape[0]}' gsm_ids with all nans for gse_id: {gse_id}")

    cpg_sites_with_all_nans = df.loc[:, df.isna().all(axis=0)]
    if cpg_sites_with_all_nans.shape[1] != 0:
        print(f"There are '{cpg_sites_with_all_nans.shape[1]}' cpg_sites with all NaNs for '{len(gse_ids)}' gse_ids.")
        # print(f"There are '{cpg_sites_with_all_nans.shape[1]}' cpg_sites with all NaNs for gse_id: {gse_id}")











    # print(df)

    fdjkfdkfdj = 1






















# &&&
