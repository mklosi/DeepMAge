from copy import deepcopy

import numpy as np
import json
import pandas as pd
from torch import nn

from modules.ml_option_3 import DeepMAgePredictor, DeepMAgeModel
from modules.ml_common import DeepMAgeBase, merge_dicts

default_args_dict = {
    "predictor_class": DeepMAgePredictor,
    "model": {
        "model_class": DeepMAgeModel,
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

    for imputation_strategy in ["median"]
    for max_epochs in [9999]
    for batch_size in [32]
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


def train_pipeline(cls, config):

    predictor = cls.new_model(config)

    df = predictor.load_data()
    train_df, test_df = predictor.split_data_by_type(df)

    # Regular train on a single fold, test, and save a model.
    predictor.train(train_df)
    _ = predictor.test_(test_df)
    predictor.save_model(cls.model_path)

    # Loading a Saved Model and test again.
    predictor = cls.load_model(cls.model_path)
    # Make sure that even if we shuffle test_df, we still get the same metrics.
    # test_df = test_df.sample(frac=1, random_state=24) # &&& does this even work?
    r = predictor.test_(test_df)
    return r

    # ## Make a prediction
    #
    # # Take the first 3 samples from test_df as new data
    # batch_sample_count = 3
    # new_data = test_df.head(batch_sample_count)
    # _, methyl_df = DeepMAgePredictor.split_df(new_data)
    #
    # # Rename the index, so it's more like actual new data.
    # # methyl_df.index = [f"sample_{i}" for i in range(batch_sample_count)]
    # methyl_df.index = [f"{gsm_id}_" for gsm_id in methyl_df.index]
    #
    # # Get reference df, so we can impute and normalize the new data (big source of contention conceptually).
    # _, ref_df = DeepMAgePredictor.split_df(df)
    #
    # # # &&&
    # # global breakpointer
    # # breakpointer = True
    #
    # prediction_df = predictor.predict_batch(methyl_df, ref_df)
    #
    # # Quick sanity check with ages from actual metadata.
    # metadata_df, _ = DeepMAgePredictor.split_df(df)
    # actual_age_df = metadata_df[[DeepMAgePredictor.age_col_str]]
    # predicted_age_df = prediction_df
    # predicted_age_df.index = [gsm_id[:-1] for gsm_id in predicted_age_df.index]
    #
    # prediction_df = actual_age_df.join(predicted_age_df, how="inner")
    # print(f"Predictions for new data:\n{prediction_df}")


def main():

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
    for config in configs:
        predictor_class = config["predictor_class"]
        result = train_pipeline(predictor_class, config)
        results.append(result)

    df = pd.DataFrame({
        "mae": [result["mae"] for result in results],
        "medae": [result["medae"] for result in results],
        "mse": [result["mse"] for result in results],
    })




    fjdkfjdkfdjk = 1





if __name__ == "__main__":
    main()
