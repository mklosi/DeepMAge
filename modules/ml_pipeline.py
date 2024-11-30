import json
from datetime import datetime

import pandas as pd

from modules.memory import Memory
from modules.ml_common import merge_dicts
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor

default_args_dict = {
    "predictor_class": "DeepMAgePredictor",
    "model": {
        "model_class": "DeepMAgeModel",
        "input_dim": 1000,
    },
}

main_args_list = [

    {
        "imputation_strategy": imputation_strategy,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr_init": lr_init,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "lr_threshold": lr_threshold,  # Bigger values make lr change faster.
        "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold,  # Bigger values make early stopping hit faster.
        "model": {
            # This is a list of integers. The len(list) shows the number of
            #   inner layers, while the number itself is the input/output
            #   neurons for each layer. For example, if the list is [128, 64, 32],
            #   it means there are 3 inner layers. The first inner layer (the 2nd actual layer)
            #   takes in how many inputs the actual first layer gives it,
            #   and outputs 128 outputs. The 2nd inner layer takes in
            #   128 inputs and gives 64 outputs. The final inner layer takes 64
            #   and gives 32. The last actual layer (final layer) would take
            #   in 32 and output 1 value, since we just want one final value.
            "inner_layers": model__inner_layers,
            "dropout": model__dropout,
            "activation_func": model__activation_func,
        },
        "loss_name": loss_name,
        "remove_nan_samples_perc": remove_nan_samples_perc,
        "test_ratio": test_ratio,
    }

    # ------------------------------------------------------

    # for imputation_strategy in ["median"]
    for imputation_strategy in ["median", "mean"]

    # for max_epochs in [9999]
    for max_epochs in [2]

    # for batch_size in [32]
    for batch_size in [32, 64]
    # for batch_size in [16, 32, 64, 128, 256, 512, 1024]

    for lr_init in [0.0001]

    for lr_factor in [0.1]

    for lr_patience in [10]

    for lr_threshold in [0.001]

    for early_stop_patience in [20]

    for early_stop_threshold in [0.0001]

    for model__input_dim in [1000]

    for model__inner_layers in [[512, 512, 256, 128]]

    for model__dropout in [0.3]

    for model__activation_func in ["elu"]

    for loss_name in ["medae"]

    for remove_nan_samples_perc in [10]

    for test_ratio in [0.2]

    # ------------------------------------------------------

    # # for imputation_strategy in ["median"]
    # for imputation_strategy in ["mean", "median"]
    #
    # for max_epochs in [999]
    # # for max_epochs in [2]
    #
    # # for batch_size in [32]
    # # for batch_size in [32, 64]
    # for batch_size in [16, 32, 64, 1024]
    #
    # for lr_init in [0.001, 0.0001, 0.00001]
    #
    # for lr_factor in [0.1, 0.5]
    #
    # for lr_patience in [10, 20, 50]
    #
    # for lr_threshold in [0.0001, 0.01, 1.0]
    #
    # for early_stop_patience in [20, 50, 100]
    #
    # for early_stop_threshold in [0.0001, 0.01, 1.0]
    #
    # for model__layer2_in in [512]
    #
    # for model__layer3_in in [512]
    #
    # for model__layer4_in in [256]
    #
    # for model__layer5_in in [128]
    #
    # for model__dropout in [0.1, 0.3, 0.5]
    #
    # for model__activation_func in ["elu", "relu"]
    #
    # for loss_name in ["medae"]
    #
    # for remove_nan_samples_perc in [10, 20]
    #
    # for test_ratio in [0.2]

]


def main():

    mem = Memory(noop=False)
    start_dt = datetime.now()

    # start_dt_iso = start_dt.isoformat()
    # result_df_path = f"result_artifacts/result_{start_dt_iso}.parquet"

    seen = set()
    main_args_json_ls = [json.dumps(dict_) for dict_ in main_args_list]
    dupes = [x for x in main_args_json_ls if x in seen or seen.add(x)]
    if dupes:
        raise ValueError(f"Duplicated main_args detected: {dupes}")

    configs = []
    for main_args_dict in main_args_list:
        merged_dict = merge_dicts(default_args_dict, main_args_dict)
        configs.append(merged_dict)

    result_df = pd.DataFrame()
    best_loss = float('inf')
    best_predictor = None
    print(f"Running '{len(configs)}' train pipelines...")
    for config in configs:
        predictor_class_name = config["predictor_class"]
        predictor = globals()[predictor_class_name](config)
        config_id = predictor.get_config_id()
        print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")

        train_pipe_dt = datetime.now()
        result = predictor.train_pipeline()
        train_pipe_runtime = datetime.now() - train_pipe_dt
        result = {"train_pipe_runtime": train_pipe_runtime, **result}
        print(f"Train pipeline runtime for '{config_id}': {train_pipe_runtime}")

        result_df_dt = datetime.now()
        # 1 Update result_df
        curr_df = pd.DataFrame([result])
        result_df = pd.concat([result_df, curr_df])
        # 2 Update result_df
        # &&&
        print(f"result_df create runtime: {datetime.now() - result_df_dt}")

        result_df = result_df.sort_values(by=predictor.config["loss_name"])
        print(f"result_df:\n{result_df.head(5)}")

        # result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
        # print(f"Saved result_df to: {result_df_path}")

        # Keep track of the best predictor so far.
        curr_loss = result[config["loss_name"]]
        if curr_loss < best_loss:
            best_loss = curr_loss
            print("Found better loss. Swapping best_predictor...")
            predictor.save()
            if best_predictor is not None:
                best_predictor.delete()
            best_predictor = predictor

    print(f"--- End of '{len(configs)}' train pipelines -----------------------------------------------------------")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
