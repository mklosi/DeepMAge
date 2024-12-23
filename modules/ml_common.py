from copy import deepcopy
from datetime import datetime
import joblib
import pandas as pd
import torch
import json
import hashlib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")


def get_df_with_cols(df, cols):
    cols_ = deepcopy(cols)
    for col in cols:
        if col not in df.columns:
            cols_.remove(col)
    df = df[cols_]
    return df


def get_config_id(config):
    if not isinstance(config, dict):
        raise ValueError(f"config needs to be of type dict, but it's instead '{type(config)}'.")

    # Remove new hyperparams with default values, in order to avoid rerunning the same tests. Backward compatibility.
    del_dict = {
        "k_folds": 1,
    }
    config = {
        key: config[key]
        for key in config
        if key not in config or key not in del_dict or config[key] != del_dict[key]
    }

    config = {key: config[key] for key in sorted(config)}
    config_json = json.dumps(config)
    config_id = hashlib.md5(config_json.encode("utf8")).hexdigest()
    return config_id


def merge_dicts(dict1, dict2):
    """
    Recursively merge dict1 and dict2, overriding values in dict1 with those in dict2
      and retaining unmatched keys from dict1.
    """
    merged = dict1.copy()  # Start with a shallow copy of dict1

    for key2, value2 in dict2.items():
        if key2 in merged and isinstance(merged[key2], dict) and isinstance(value2, dict):
            merged[key2] = merge_dicts(merged[key2], value2)
        elif key2 in merged and (isinstance(merged[key2], dict) and not isinstance(value2, dict)):
            raise ValueError(f"Key '{key2}' refers to a dict in dict1, but not in dict2")
        elif key2 in merged and (not isinstance(merged[key2], dict) and isinstance(value2, dict)):
            raise ValueError(f"Key '{key2}' refers to a dict in dict2, but not in dict1")
        else:
            merged[key2] = value2

    return merged


class MethylationDataset(Dataset):
    def __init__(self, features, ages):
        self.features = features
        self.ages = ages

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": torch.tensor(self.features.values[idx], dtype=torch.float32)
        }
        if self.ages is not None:
            sample["age"] = torch.tensor(self.ages.values[idx], dtype=torch.float32)
        return sample


class DeepMAgeBase: # TODO do we even need this?
    pass

    # @classmethod
    # def load_model(cls, model_path):
    #     raise NotImplementedError()
    #
    # @classmethod # TODO not needed.
    # def new_model(cls):
    #     raise NotImplementedError()
    #
    # def save_model(self, save_path):
    #     raise NotImplementedError()
    #
    # @staticmethod
    # def load_data(file_path):
    #     raise NotImplementedError()
    #
    # def prepare_features(self, df, is_training):
    #     raise NotImplementedError()
    #
    # @staticmethod
    # def split_data(features, ages, test_size=0.2):
    #     raise NotImplementedError()
    #
    # def train(self, train_loader, val_loader, epochs):
    #     raise NotImplementedError()
    #
    # def train_with_cross_validation(self, data, folds):
    #     raise NotImplementedError()
    #
    # def validate(self, val_loader):
    #     raise NotImplementedError()
    #
    # def test(self, test_loader):
    #     raise NotImplementedError()
    #
    # def predict_batch(self, features):
    #     raise NotImplementedError()
    #
    # @classmethod
    # def training_pipeline(cls):
    #     raise NotImplementedError()
