import numpy as np
from torch import nn

from modules.ml_option_3 import DeepMAgePredictor, DeepMAgeModel
from modules.ml_common import DeepMAgeBase

pipeline_configs = [
    {
        "predictor_class": DeepMAgePredictor,
        "imputation_strategy": "median",
        "max_epochs": 9999,
        "batch_size": 32,
        "lr_init": 0.0001,
        "lr_factor": 0.1,
        "lr_patience": 10,
        "lr_threshold": 0.001,  # Bigger values make lr change faster.
        "early_stop_patience": 20,
        "early_stop_threshold": 0.0001,  # Bigger values make early stopping hit faster.
        "model": {
            "model_class": DeepMAgeModel,
            "input_dim": 1000,
            "layer2_in": 512,
            "layer3_in": 512,
            "layer4_in": 256,
            "layer5_in": 128,
            "dropout": 0.3,
            "activation_func": nn.ELU(),
        },
        "loss_name": "medae",
        "remove_nan_samples_perc": 10,
        "test_ratio": 0.2,
    },
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
    test_df = test_df.sample(frac=1, random_state=24) # &&& does this even work?
    _ = predictor.test_(test_df)

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
    for config in pipeline_configs:
        predictor_class = config["predictor_class"]
        train_pipeline(predictor_class, config)


if __name__ == "__main__":
    main()
