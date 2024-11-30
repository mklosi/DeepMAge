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
    },
}

main_args_list = [

    {
        # "predictor_class": DeepMAgePredictor,
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
            # "model_class": DeepMAgeModel,
            "input_dim": model__input_dim,
            "layer2_in": model__layer2_in,
            "layer3_in": model__layer3_in,
            "layer4_in": model__layer4_in,
            "layer5_in": model__layer5_in,
            "dropout": model__dropout,
            "activation_func": model__activation_func,
        },
        "loss_name": loss_name,
        "remove_nan_samples_perc": remove_nan_samples_perc,
        "test_ratio": test_ratio,
    }

    # for imputation_strategy in ["median"]
    for imputation_strategy in ["mean", "median"]

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

    for model__layer2_in in [512]

    for model__layer3_in in [512]

    for model__layer4_in in [256]

    for model__layer5_in in [128]

    for model__dropout in [0.3]

    for model__activation_func in ["elu"]

    for loss_name in ["medae"]

    for remove_nan_samples_perc in [10]

    for test_ratio in [0.2]

]


def main():

    mem = Memory(noop=False)
    start_dt = datetime.now()

    seen = set()
    main_args_json_ls = [json.dumps(dict_) for dict_ in main_args_list]
    dupes = [x for x in main_args_json_ls if x in seen or seen.add(x)]
    if dupes:
        raise ValueError(f"Duplicated main_args detected: {dupes}")

    configs = []
    for main_args_dict in main_args_list:
        merged_dict = merge_dicts(default_args_dict, main_args_dict)
        configs.append(merged_dict)

    results = []
    best_predictor = None
    best_loss = float('inf')
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

        # Keep track of the best predictor so far.
        curr_loss = result[config["loss_name"]]
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_predictor = predictor

        results.append(result)

    print(f"--- End of '{len(configs)}' train pipelines -----------------------------------------------------------")

    # Save best predictor
    print("Saving best_predictor...")
    best_predictor.save_predictor()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by=best_predictor.config["loss_name"])
    print(f"result_df:\n{result_df}")

    start_dt_str = start_dt.isoformat()
    result_df_path = f"result_artifacts/result_{start_dt_str}.parquet"
    result_df.to_parquet(result_df_path, engine='pyarrow', index=False)
    print(f"Saved result_df to: {result_df_path}")

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
