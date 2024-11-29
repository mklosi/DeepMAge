import numpy as np
from torch import nn

from modules.ml_option_3 import DeepMAgePredictor, DeepMAgeModel
from modules.ml_common import DeepMAgeBase

some_name = [
    {
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
        "predictor_class": DeepMAgePredictor,
        "imputation_strategy": "median",
        "max_epochs": 9999,
        "batch_size": 32,


    },



]












def main():
    pass





if __name__ == "__main__":
    main()
