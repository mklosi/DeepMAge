import pandas as pd
from memory import Memory
from datetime import datetime

gsm_id_str = "gsm_id"
cpg_site_id_str = "cpg_site_id"
metadata_parquet_path = "resources/metadata.parquet"


def rename_cols(df, mapping, key):
    df = df.rename(columns=mapping)
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()

    df = df.set_index(gsm_id_str).add_prefix(key)
    df.index.name = gsm_id_str

    return df


def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.min_rows', 100)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # col mapping:
    mapping = {
        "GEO_RUN": "gsm_id",
        "geo_run": "gsm_id",
        "Actual age, yrs": "actual_age_years",
        "Predicted age, yrs": "predicted_age_years",
        "log(Horvath)_full": "log_horvath_full",
        "log(Horvath)_shrunk": "log_horvath_shrunk",
    }

    mem = Memory(noop=True)
    dt_start = datetime.now()

    mem.log_memory(print, "before___m")

    de_novo_path = "resources/de_novo_linear_model_verification_predictions.txt"
    deepmage_partition_path = "resources/DeepMAge_data_partition.txt"
    deepmage_predictions_path = "resources/DeepMAge_predictions.txt"
    original_353_predictions_path = "resources/original_353_clock_predictions.txt"

    de_novo_df = rename_cols(
        pd.read_csv(de_novo_path, sep='\t'), mapping, "1."
    )
    deepmage_partition_df = rename_cols(
        pd.read_csv(deepmage_partition_path, sep='\t'), mapping, "2."
    )
    deepmage_predictions_df = rename_cols(
        pd.read_csv(deepmage_predictions_path, sep='\t'), mapping, "3."
    )
    original_353_predictions_df = rename_cols(
        pd.read_csv(original_353_predictions_path, sep='\t'), mapping, "4."
    )

    mem.log_memory(print, "after_load")

    df = (
        de_novo_df
        .merge(deepmage_partition_df, on=gsm_id_str, how='outer')
        .merge(deepmage_predictions_df, on=gsm_id_str, how='outer')
        .merge(original_353_predictions_df, on=gsm_id_str, how='outer')
    )

    mem.log_memory(print, "aftermerge")

    df['row_number'] = range(len(df))
    # Move row_number to the beginning
    df = df[['row_number'] + [col for col in df.columns if col != 'row_number']]
    df = df.sort_index()

    mem.log_memory(print, "after_sort")

    df.to_parquet(metadata_parquet_path, engine='pyarrow', index=True)

    mem.log_memory(print, "after_save")
    print(f"Total runtime: {datetime.now() - dt_start}")


if __name__ == '__main__':
    main()
