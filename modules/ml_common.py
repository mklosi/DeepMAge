from datetime import datetime
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")


class MethylationDataset(Dataset):
    def __init__(self, features, ages, is_training=True):  # &&& is this correctly named? &&& do we even need it. we can just say `if self.ages...
        self.features = features.values
        self.ages = ages.values
        self.is_training = is_training

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            "features": torch.tensor(self.features[idx], dtype=torch.float32)
        }
        if self.is_training:
            sample["age"] = torch.tensor(self.ages[idx], dtype=torch.float32)
        return sample


class DeepMAgeBase: # &&& do we even need this?
    pass

    # @classmethod
    # def load_model(cls, model_path):
    #     raise NotImplementedError()
    #
    # @classmethod
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
