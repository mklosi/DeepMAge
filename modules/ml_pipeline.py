import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from modules.memory import Memory
from modules.ml_common import merge_dicts, get_config_id
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor, set_seeds

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# default_loss_name = "mae"
default_loss_name = "medae"

default_args_dict = {
    "predictor_class": "DeepMAgePredictor",
    "model": {
        "model_class": "DeepMAgeModel",
        "input_dim": 1000,
    },
    "loss_name": default_loss_name,
}

main_args_list = [

    {
        "imputation_strategy": imputation_strategy,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr_init": lr_init,
        "weight_decay": weight_decay,  # Add L2 regularization.
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "lr_threshold": lr_threshold,  # Bigger values make lr change faster.
        "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold,  # Bigger values make early stopping hit faster.
        "model": {

            # "layer2_in": model__layer2_in,
            # "layer3_in": model__layer3_in,
            # "layer4_in": model__layer4_in,
            # "layer5_in": model__layer5_in,

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
        "remove_nan_samples_perc": remove_nan_samples_perc,
        "test_ratio": test_ratio,
    }

    # ------------------------------------------------------

    # for imputation_strategy in ["median"]
    #
    # # for max_epochs in [9999]
    # # for max_epochs in [2]
    # for max_epochs in [2, 3]
    # # for max_epochs in [5, 4, 3, 2]
    #
    # # for batch_size in [32]
    # for batch_size in [64, 32]
    #
    # for lr_init in [0.0001]
    #
    # for weight_decay in [0.0]
    #
    # for lr_factor in [0.1]
    #
    # for lr_patience in [10]
    #
    # for lr_threshold in [0.001]
    #
    # for early_stop_patience in [20]
    #
    # for early_stop_threshold in [0.0001]
    #
    # for model__input_dim in [1000]
    #
    # # for model__layer2_in in [512]
    # # for model__layer3_in in [512]
    # # for model__layer4_in in [256]
    # # for model__layer5_in in [128]
    #
    # for model__inner_layers in [[512, 512, 256, 128]]
    #
    # for model__dropout in [0.3]
    #
    # for model__activation_func in ["elu"]
    #
    # for remove_nan_samples_perc in [10]
    #
    # for test_ratio in [0.2]

    # ------------------------------------------------------

    for model__inner_layers in [
        [512, 512, 256, 128],
        [512, 512, 256, 256, 128, 64],
        [1024, 512, 256, 128, 64, 32, 16, 8],
    ]

    # for imputation_strategy in ["median"]
    for imputation_strategy in ["mean", "median"]

    for max_epochs in [999]
    # for max_epochs in [2]

    # for batch_size in [32]
    # for batch_size in [32, 64]
    for batch_size in [16, 32, 64]

    for lr_init in [0.001, 0.00001]

    for weight_decay in [0.0]

    for lr_factor in [0.1, 0.5]

    for lr_patience in [10, 50]

    for lr_threshold in [0.001, 1.0]

    for early_stop_patience in [30, 100]

    for early_stop_threshold in [0.001, 1.0]

    for model__dropout in [0.1, 0.3]

    for model__activation_func in ["elu", "relu"]

    for remove_nan_samples_perc in [10, 30]

    for test_ratio in [0.2]

    # ------------------------------------------------------

    # for model__inner_layers in [
    #     [512, 512, 512, 512, 512],
    # ]
    #
    # # for imputation_strategy in ["median"]
    # for imputation_strategy in ["median"]
    #
    # for max_epochs in [999]
    # # for max_epochs in [2]
    #
    # # for batch_size in [32]
    # # for batch_size in [32, 64]
    # for batch_size in [32]
    #
    # for lr_init in [0.0001]
    #
    # for weight_decay in [1e-3]
    #
    # for lr_factor in [0.1]
    #
    # for lr_patience in [25]
    #
    # for lr_threshold in [0.01]
    #
    # for early_stop_patience in [50]
    #
    # for early_stop_threshold in [0.01]
    #
    # for model__dropout in [0.3]
    #
    # for model__activation_func in ["elu"]
    #
    # for remove_nan_samples_perc in [20]
    #
    # for test_ratio in [0.2]

]


def main(override, overwrite):

    mem = Memory(noop=False)
    start_dt = datetime.now()

    result_df_path = Path(f"result_artifacts/result_df_temp.parquet")

    seen = set()
    main_args_json_ls = [json.dumps(dict_) for dict_ in main_args_list]
    dupes = [x for x in main_args_json_ls if x in seen or seen.add(x)]
    if dupes:
        raise ValueError(f"Duplicated main_args detected: {dupes}")

    configs = []
    for main_args_dict in main_args_list:
        merged_dict = merge_dicts(default_args_dict, main_args_dict)
        configs.append(merged_dict)

    if result_df_path.exists() and not overwrite:
        result_df = pd.read_parquet(result_df_path)
        best_ser = result_df.sort_values(by=default_loss_name).iloc[0]
        best_loss = best_ser[default_loss_name]
        config = json.loads(best_ser["config"])
        predictor_class_name = config["predictor_class"]
        predictor_class = globals()[predictor_class_name]
        best_predictor = predictor_class(config)
    else:
        result_df = pd.DataFrame()
        best_loss = float('inf')
        best_predictor = None

    print(f"Found '{len(configs)}' total configs.")
    configs = [config for config in configs if override or get_config_id(config) not in result_df.index]
    print(f"Running '{len(configs)}' new train pipelines...")

    # &&& keep these for filtering.
    config_ids_for_filtering = []

    for config in configs:
        set_seeds()  # Reset the seeds for reproducibility.
        config_id = get_config_id(config)

        print(f"--- Train pipeline for config_id: {config_id} --------------------------------------------")
        print(f"config:\n{json.dumps(config, indent=4)}")
        predictor_class_name = config["predictor_class"]
        predictor_class = globals()[predictor_class_name]
        predictor = predictor_class(config)

        train_pipe_dt = datetime.now()
        result_dict = predictor.train_pipeline()
        pipeline_runtime = datetime.now() - train_pipe_dt
        print(f"Train pipeline runtime for '{config_id}': {pipeline_runtime}")

        # # &&&
        # predictor.save()

        # Keep track of the best predictor so far.
        curr_loss = result_dict[config["loss_name"]]
        if curr_loss < best_loss:
            best_loss = curr_loss
            print("Found better loss. Swapping best_predictor...")

            predictor.save()
            if best_predictor is not None:
                best_predictor.delete()

            best_predictor = predictor

        result_dict = {
            "config_id": config_id,
            "start_dt": start_dt,
            "pipeline_runtime": pipeline_runtime,
            **result_dict,
            "config": json.dumps(config),
        }
        curr_df = pd.DataFrame([result_dict]).set_index("config_id")
        result_df = pd.concat([result_df, curr_df])
        result_df = result_df.loc[~result_df.index.duplicated(keep='last')]

        # &&&
        config_ids_for_filtering.append(config_id)
        result_df = result_df[result_df.index.isin(config_ids_for_filtering)]

        result_df = result_df.sort_values(by=default_loss_name)
        print(f"result_df:\n{result_df.drop(columns='config')}")

        result_df.to_parquet(result_df_path, engine='pyarrow', index=True)
        print(f"Saved result_df to: {result_df_path}")

    print(f"--- End of '{len(configs)}' train pipelines -----------------------------------------------------------")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main(override=False, overwrite=False)
