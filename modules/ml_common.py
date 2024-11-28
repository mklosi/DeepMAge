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
    """Dataset class for DNA methylation data."""
    def __init__(self, features, ages, is_training=True):
        self.features = features
        self.ages = ages
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


class DeepMAgeBase:
    """Base class for DeepMAge implementations."""
    model_path = "model_artifacts/deepmage_model.pth"

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None

    @staticmethod
    def load_data(file_path):
        """Load and preprocess the methylation data."""
        df = pd.read_parquet(file_path)
        return df

    def prepare_features(self, df, is_training=True):
        """Normalize features and split data."""
        features = df.drop(columns=["age"]).values
        if is_training:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        ages = df["age"].values if "age" in df.columns else None
        return features, ages

    @staticmethod
    def split_data(features, ages, test_size=0.2):
        """Split data into training and validation sets."""
        n_samples = len(features)
        split_idx = int(n_samples * (1 - test_size))
        return (
            features[:split_idx], ages[:split_idx],
            features[split_idx:], ages[split_idx:]
        )

    @classmethod
    def load_model(cls, model_path):
        """Load a saved model."""
        raise NotImplementedError()

    def save_model(self, save_path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def train(self, train_loader, val_loader, epochs):
        raise NotImplementedError()

    def test(self, test_loader):
        raise NotImplementedError()

    def predict_batch(self, features):
        raise NotImplementedError()
