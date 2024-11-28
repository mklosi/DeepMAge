from collections import defaultdict
import random
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd

from modules.memory import Memory
from modules.ml_common import DeepMAgeBase, MethylationDataset, DataLoader
from datetime import datetime

mem = Memory(noop=False)

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


def set_seeds(seed=42):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)  # Only this one actually matters (empirically), but leaving the others for reference.
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)


set_seeds()


class DeepMAgeModel(nn.Module):
    """Deep neural network for age prediction."""
    def __init__(self, input_dim):
        super(DeepMAgeModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.dropout(x)
        x = self.elu(self.fc3(x))
        x = self.dropout(x)
        x = self.elu(self.fc4(x))
        x = self.fc5(x)
        return x


class MedAELoss(nn.Module):
    def __init__(self):
        super(MedAELoss, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, predictions, targets):
        absolute_errors = torch.abs(predictions - targets)
        medae = torch.median(absolute_errors)
        return medae


class DeepMAgePredictor(DeepMAgeBase):

    model_path = "model_artifacts/model_option_3.predictor"
    metadata_df_path = "resources/metadata_derived.parquet"
    methyl_df_path = "resources/methylation_data.parquet"

    age_col_str = "actual_age_years"
    metadata_cols_of_interest = ["gse_id", "type", age_col_str] # &&& can I be reminded of the benefit of doing cls vs. self?

    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))  #$ hyper
        self.input_dim = 1000
        self.batch_size = 32
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = DeepMAgeModel(input_dim=self.input_dim).to(self.device)

        self.criterions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),  # Computes Mean Absolute Error (MAE)
            "medae": MedAELoss(),  # Computes Median Absolute Error (MedAE)
        }
        self.loss_str = "mae"

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    @classmethod
    def load_model(cls, save_path):
        """Load the entire DeepMAgePredictor object."""
        with open(save_path, 'rb') as f:
            instance = joblib.load(f)
        print(f"'{cls.__name__}' loaded from: {save_path}")
        return instance

    @classmethod
    def new_model(cls):
        return cls()

    def save_model(self, save_path):
        """Save the entire DeepMAgePredictor object."""
        with open(save_path, 'wb') as f:
            joblib.dump(self, f)
        print(f"'{self.__class__.__name__}' saved to: {save_path}")

    @classmethod
    def join_dfs(cls, metadata_df, methyl_df):
        df = methyl_df.join(metadata_df, how="inner")
        ## Rearrange cols, so the metadata cols are first.
        cols = list(df.columns)
        for i, col in enumerate(cls.metadata_cols_of_interest):
            cols.insert(i, cols.pop(cols.index(col)))

        df = df[cols]
        return df

    @classmethod
    def split_df(cls, df):
        metadata_df = df[cls.metadata_cols_of_interest]
        methyl_df = df.drop(columns=cls.metadata_cols_of_interest)
        return metadata_df, methyl_df

    @classmethod
    def load_data(cls):
        metadata_df = pd.read_parquet(cls.metadata_df_path)
        methyl_df = pd.read_parquet(cls.methyl_df_path)

        cols_to_drop = sorted(set(metadata_df.columns.tolist()) - set(cls.metadata_cols_of_interest))
        metadata_df = metadata_df.drop(columns=cols_to_drop)

        df = cls.join_dfs(metadata_df, methyl_df)

        df = df[df["type"].isin(["train", "verification"])]

        # &&&
        ## Show which gsm_ids have missing values and the corresponding cpg sites that contain those nans.
        nan_counts = df.isna().sum(axis=1)  # Count NaNs per GSM ID
        gsm_ids_with_nans = nan_counts[nan_counts > 0]
        nan_cpg_sites = df.apply(lambda row: ','.join(row.index[row.isna()]), axis=1)
        gsm_ids_with_nans_df = pd.DataFrame({
            'gsm_id': gsm_ids_with_nans.index,
            'gse_id': df.loc[gsm_ids_with_nans.index, 'gse_id'].values,
            'type': df.loc[gsm_ids_with_nans.index, 'type'].values,
            'nan_count': gsm_ids_with_nans.values,
            'cpg_sites_with_nan': nan_cpg_sites[gsm_ids_with_nans.index].values
        })
        gsm_ids_with_nans_sorted_df = gsm_ids_with_nans_df.sort_values(by="nan_count", ascending=False)

        # &&&
        ## Show which cpg site ids have missing values and the corresponding gsm_ids that contain those nans.
        nan_counts_per_cpg = df.isna().sum(axis=0)
        cpg_sites_with_nans = nan_counts_per_cpg[nan_counts_per_cpg > 0]
        nan_gsm_ids_per_cpg = df.apply(lambda col: ','.join(df.index[col.isna()]), axis=0)
        cpg_sites_with_nans_df = pd.DataFrame({
            'cpg_site_id': cpg_sites_with_nans.index,
            'nan_count': cpg_sites_with_nans.values,
            'gsm_ids_with_nan': nan_gsm_ids_per_cpg[cpg_sites_with_nans.index].values
        })
        cpg_sites_with_nans_sorted_df = cpg_sites_with_nans_df.sort_values(by="nan_count", ascending=False)

        return df

    def normalize_group(self, df):
        metadata_df, methyl_df = self.split_df(df)
        methyl_df = pd.DataFrame(self.scaler.fit_transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns)
        df = self.join_dfs(metadata_df, methyl_df)
        return df

    def prepare_features(self, df, is_training=True): # &&& this is correctly named.

        ## Remove samples with more than X nan values across all cpg sites.
        nan_counts = df.isna().sum(axis=1)  # Count NaNs per GSM ID
        ten_perc = self.input_dim // 10 #$ hyper (and all others...)
        gsm_ids_with_nans = nan_counts[nan_counts > ten_perc].index
        df = df[~df.index.isin(gsm_ids_with_nans)]

        metadata_df, methyl_df = self.split_df(df)

        ## impute #$ docs
        if is_training:
            methyl_df = pd.DataFrame(self.imputer.fit_transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns)
        else:
            methyl_df = pd.DataFrame(self.imputer.transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns)

        ## normalize #$ docs
        df = self.join_dfs(metadata_df, methyl_df)

        # # filter
        # df = df[df["gse_id"] == "GSE84624"]

        df = df.groupby('gse_id', group_keys=False).apply(self.normalize_group)

        # # checker
        # min_max_df = (
        #     methyl_df
        #     .agg(['min', 'max'])  # Compute min and max for each column
        #     .T  # Transpose for a tabular structure
        #     .reset_index()  # Convert index to column
        # )
        # min_max_df.columns = ['cpg_site_id', 'min_value', 'max_value']
        # min_max_df = min_max_df.sort_values(by='cpg_site_id').reset_index(drop=True)

        metadata_df, methyl_df = self.split_df(df)
        features = methyl_df.values

        # &&& metadata_df could be null during prediction???
        # &&& do we need to scale age?
        ages = metadata_df[self.age_col_str].values if self.age_col_str in metadata_df.columns else None

        return features, ages

    @staticmethod
    def split_data_by_type(df):
        train_df = df[df["type"] == "train"]
        test_df = df[df["type"] == "verification"]
        return train_df, test_df

    @staticmethod
    def split_data_by_percent(df):
        return train_test_split(df, test_size=0.2, random_state=42)

    def train(self, train_df):
        print("Training model...")
        train_dt = datetime.now()

        epochs = 2  #$ hyper. init: 100

        train_data, val_data = self.split_data_by_percent(train_df)

        features_train, ages_train = self.prepare_features(train_data, is_training=True) # &&& need to play with these more.
        features_val, ages_val = self.prepare_features(val_data, is_training=False)

        train_dataset = MethylationDataset(features_train, ages_train, is_training=True)
        val_dataset = MethylationDataset(features_val, ages_val, is_training=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            print(f"--- Epoch '{epoch + 1}/{epochs}' --------------------")
            self.model.train()
            epoch_loss = 0
            for i, batch in enumerate(train_loader):
                print(f"Processing batch: {i+1}/{len(train_loader)}")

                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features).squeeze()
                loss = self.criterions[self.loss_str](predictions, ages)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            val_loss = self.validate(val_loader)
            print(f"Epoch '{epoch+1}/{epochs}', Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}")

        mem.log_memory(print, "train")
        print(f"train runtime: {datetime.now() - train_dt}")

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)
                predictions = self.model(features).squeeze()
                loss = self.criterions[self.loss_str](predictions, ages)
                total_loss += loss.item()
        val_loss = total_loss / len(val_loader)
        return val_loss

    def test_(self, test_df): # &&& rename after.
        print("Testing model...")

        # &&& play with these. possible rename for one/more of theis_training.
        features_test, ages_test = self.prepare_features(test_df, is_training=False)
        test_dataset = MethylationDataset(features_test, ages_test, is_training=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        metric_results = {}
        actual_ages_all = []
        predictions_all = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                predictions = self.model(features).squeeze()
                ages = batch["age"].to(self.device)

                predictions_all.extend(predictions.cpu())
                actual_ages_all.extend(ages.cpu())

            predictions_all = torch.tensor(predictions_all)
            actual_ages_all = torch.tensor(actual_ages_all)

            for name, criterion in self.criterions.items():
                metric_results[name] = criterion(predictions_all, actual_ages_all).item()

        for name, value in metric_results.items():
            print(f"Test '{name}': {value}")

        return metric_results

    def predict_batch(self, new_data):

        self.model.eval()

        features, _ = self.prepare_features(new_data, is_training=False)
        dataset = MethylationDataset(features, None, is_training=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []
        gsm_ids = new_data.index.tolist()  # Preserve gsm_id for output

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                batch_predictions = self.model(features).squeeze()
                predictions.extend(batch_predictions.cpu().numpy())

        # Format predictions into a DataFrame for easier usage
        predictions_df = pd.DataFrame({
            "gsm_id": gsm_ids,
            "predicted_age": predictions
        }).set_index("gsm_id")

        return predictions_df

    @classmethod
    def train_pipeline(cls):

        predictor = cls.new_model()

        df = predictor.load_data()
        train_df, test_df = predictor.split_data_by_type(df)

        # Regular train on a single fold, test, and save a model.
        predictor.train(train_df)
        _ = predictor.test_(test_df)
        predictor.save_model(cls.model_path)

        # Loading a Saved Model and test again.
        predictor = cls.load_model(cls.model_path)
        # test_df = test_df.sample(frac=1, random_state=43)
        _ = predictor.test_(test_df)

        # ## Make a prediction
        #
        # # Take the first 3 samples from test_df as new data
        # new_data = test_df.head(3)
        # _, methyl_df = DeepMAgePredictor.split_df(new_data)
        # predictions = predictor.predict_batch(methyl_df)
        # print(f"Predictions for new data:\n{predictions}")




        fdjkfd = 1



if __name__ == "__main__":
    DeepMAgePredictor.train_pipeline()
