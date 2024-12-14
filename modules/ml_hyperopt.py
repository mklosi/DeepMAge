import json
from datetime import datetime
from pathlib import Path
from hyperopt import hp
import pandas as pd
from hyperopt import Trials
from modules.memory import Memory
from modules.ml_common import merge_dicts, get_config_id
# noinspection PyUnresolvedReferences
from modules.ml_option_3 import DeepMAgePredictor, set_seeds

from hyperopt import fmin, tpe, STATUS_OK
from modules.ml_common import merge_dicts, get_config_id

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

default_loss_name = "medae"

default_args_dict = {
    "predictor_class": "DeepMAgePredictor",
    "model": {
        "model_class": "DeepMAgeModel",
        "input_dim": 1000,
    },
    "loss_name": default_loss_name,
}

search_space = {
    "imputation_strategy": hp.choice("imputation_strategy", ["median"]),

    "max_epochs": hp.choice("max_epochs", [2]),
    # "max_epochs": hp.choice("max_epochs", [2, 3]),

    "batch_size": hp.choice("batch_size", [32]),
    # "batch_size": hp.choice("batch_size", [32, 64]),

    "lr_init": hp.choice("lr_init", [0.0001]),

    "lr_factor": hp.choice("lr_factor", [0.1]),

    "lr_patience": hp.choice("lr_patience", [10]),

    "lr_threshold": hp.choice("lr_threshold", [0.001]), # &&& try without list.

    "early_stop_patience": hp.choice("early_stop_patience", [20]),

    "early_stop_threshold": hp.choice("early_stop_threshold", [0.0001]),

    "model": {
        "hidden_edges": hp.choice(
            "hidden_edges", [
                [512, 512, 256, 128],
            ]
        ),

        "dropout": hp.choice("dropout", [0.3]),

        "activation_func": hp.choice("activation_func", ["elu"]),
    },

    "remove_nan_samples_perc_2": hp.choice("remove_nan_samples_perc_2", [10]),

    "test_ratio": hp.choice("test_ratio", [0.2]),
}


def objective(hyperparams):
    # Combine default arguments with the sampled hyperparameters
    config = merge_dicts(default_args_dict, hyperparams)

    # Initialize the predictor
    set_seeds()  # Reset seeds for reproducibility
    predictor_class_name = config["predictor_class"]
    predictor_class = globals()[predictor_class_name]
    predictor = predictor_class(config)

    # Train the model and get the result
    result_dict = predictor.train_pipeline()
    loss = result_dict[config["loss_name"]]

    # Hyperopt requires a specific return format
    return {
        "loss": loss,  # Metric to minimize
        "status": STATUS_OK,  # Trial status
        "result_dict": result_dict,  # Extra information for debugging
        "config": config,  # Save config for analysis
    }


def main():

    mem = Memory(noop=False)
    start_dt = datetime.now()

    trials = Trials()

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=999, #  ??
        timeout=86400,  # 1 day
        loss_threshold=2.77,
        trials=trials,
        # &&& rstate i don't like this.
        allow_trials_fmin=True, # default.
        pass_expr_memo_ctrl=False,#default . check what this is.
        catch_eval_exceptions=False,#default . check what this is.
        verbose=True,#default
        # return_argmin=True,#I know this should be True.
        # points_to_evaluate=None,# I guess I can't even use this.
        # max_queue_len=1, #  check what this is.
        show_progressbar=True,
        # early_stop_fn=None,
        # trials_save_file="", # &&& add here.
    )

    # Display the best hyperparameters
    print("Best hyperparameters:")
    print(best)

    # Analyze results
    best_trial = trials.best_trial
    print("Best trial:")
    print(best_trial)

    mem.log_memory(print, "ml_pipeline_end")
    print(f"Total runtime: {datetime.now() - start_dt}")


if __name__ == "__main__":
    main()
