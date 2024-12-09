from collections import defaultdict
import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

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
from modules.ml_common import DeepMAgeBase, MethylationDataset, DataLoader, get_config_id
from datetime import datetime

mem = Memory(noop=False)

pd.set_option('display.max_columns', 8)
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

breakpointer = False


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Only this one actually matters (empirically), but leaving the others for reference.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)


set_seeds()


class DeepMAgeModel(nn.Module):
    """Deep neural network for age prediction."""

    def __init__(self, config):
        super(DeepMAgeModel, self).__init__()
        self.config = config

        inner_layers = self.config["model.inner_layers"]
        if isinstance(inner_layers, str):
            inner_layers = json.loads(inner_layers)

        # Dynamically create layers.
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.config["model.input_dim"], inner_layers[0]))
        previous_out = inner_layers[0]
        for inner_layer in inner_layers[1:]:
            self.fcs.append(nn.Linear(previous_out, inner_layer))
            previous_out = inner_layer
        self.fcs.append(nn.Linear(previous_out, 1))

        self.dropout = nn.Dropout(self.config["model.dropout"])
        activation_funcs = {
            "celu": nn.CELU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        self.activation_func = activation_funcs[self.config["model.activation_func"]]

    def forward(self, x):
        for i, fc in enumerate(self.fcs[:-1]):
            x = self.activation_func(fc(x))
            # Add dropout only if not the final hidden layer
            if i < len(self.fcs[:-1]) - 1:
                x = self.dropout(x)
        x = self.fcs[-1](x)
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

    predictor_base_dir = "predictor_artifacts"
    metadata_df_path = "resources/metadata_derived.parquet"
    methyl_df_path = "resources/methylation_data.parquet"

    age_col_str = "actual_age_years"
    metadata_cols = ["gse_id", "type", age_col_str] # &&& can I be reminded of the benefit of doing cls vs. self?

    def __init__(self, config):
        self.config = config
        self.config_id = get_config_id(self.config)
        self.normalization_strategy = self.config["normalization_strategy"]
        self.split_train_test_by_percent = self.config["split_train_test_by_percent"]
        self.k_folds = self.config["k_folds"]

        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else
        #     "mps" if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        self.device = "cpu"

        self.imputer = SimpleImputer(strategy=self.config["imputation_strategy"])
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        model_class_name = config["model.model_class"]
        model_class = globals()[model_class_name]
        self.model = model_class(config=self.config).to(self.device)
        self.criterions = {
            "mse": nn.MSELoss(),
            "mae": nn.L1Loss(),  # Computes Mean Absolute Error (MAE)
            "medae": MedAELoss(),  # Computes Median Absolute Error (MedAE)
        }
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["lr_init"],
            weight_decay=self.config["weight_decay"],  # 0.0 is default.
        )

    def load(self):
        """Load the entire DeepMAgePredictor object."""

        predictor_path = f"{self.predictor_base_dir}/{self.config_id}.predictor"
        with open(predictor_path, 'rb') as f:
            state = joblib.load(f)
        self.__dict__.update(state["predictor_state"].__dict__)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        print(f"'{self.__class__.__name__}' loaded from: {predictor_path}")

    def save(self):
        """Save the entire DeepMAgePredictor object."""

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "predictor_state": self,
        }
        predictor_path = f"{self.predictor_base_dir}/{self.config_id}.predictor"
        with open(predictor_path, 'wb') as f:
            joblib.dump(state, f)
        print(f"'{self.__class__.__name__}' saved to: {predictor_path}")

    def delete(self):
        """Delete the saved artifact for this predictor to save money."""
        predictor_path = Path(f"{self.predictor_base_dir}/{self.config_id}.predictor")
        predictor_path.unlink(missing_ok=True)
        print(f"Deleted predictor at: {predictor_path}")

    @staticmethod
    def join_dfs(metadata_df, methyl_df):
        meta_cols = metadata_df.columns
        df = methyl_df.join(metadata_df, how="inner")
        # Rearrange cols, so the metadata cols are first.
        df = df[[*meta_cols, *methyl_df]]
        return df

    @classmethod
    def split_df(cls, df):
        meta_cols_curr = sorted(set(df.columns).intersection(set(cls.metadata_cols)))
        metadata_df = df[meta_cols_curr]
        methyl_df = df.drop(columns=meta_cols_curr)
        return metadata_df, methyl_df

    @classmethod
    def load_data(cls):
        metadata_df = pd.read_parquet(cls.metadata_df_path)
        methyl_df = pd.read_parquet(cls.methyl_df_path)

        cols_to_drop = sorted(set(metadata_df.columns.tolist()) - set(cls.metadata_cols))
        metadata_df = metadata_df.drop(columns=cols_to_drop)

        df = cls.join_dfs(metadata_df, methyl_df)

        df = df[df["type"].isin(["train", "verification"])]

        # # &&&
        # ## Show which gsm_ids have missing values and the corresponding cpg sites that contain those nans.
        # nan_counts = df.isna().sum(axis=1)  # Count NaNs per GSM ID
        # gsm_ids_with_nans = nan_counts[nan_counts > 0]
        # nan_cpg_sites = df.apply(lambda row: ','.join(row.index[row.isna()]), axis=1)
        # gsm_ids_with_nans_df = pd.DataFrame({
        #     'gsm_id': gsm_ids_with_nans.index,
        #     'gse_id': df.loc[gsm_ids_with_nans.index, 'gse_id'].values,
        #     'type': df.loc[gsm_ids_with_nans.index, 'type'].values,
        #     'nan_count': gsm_ids_with_nans.values,
        #     'cpg_sites_with_nan': nan_cpg_sites[gsm_ids_with_nans.index].values
        # })
        # gsm_ids_with_nans_sorted_df = gsm_ids_with_nans_df.sort_values(by="nan_count", ascending=False)
        #
        # # &&&
        # ## Show which cpg site ids have missing values and the corresponding gsm_ids that contain those nans.
        # nan_counts_per_cpg = df.isna().sum(axis=0)
        # cpg_sites_with_nans = nan_counts_per_cpg[nan_counts_per_cpg > 0]
        # nan_gsm_ids_per_cpg = df.apply(lambda col: ','.join(df.index[col.isna()]), axis=0)
        # cpg_sites_with_nans_df = pd.DataFrame({
        #     'cpg_site_id': cpg_sites_with_nans.index,
        #     'nan_count': cpg_sites_with_nans.values,
        #     'gsm_ids_with_nan': nan_gsm_ids_per_cpg[cpg_sites_with_nans.index].values
        # })
        # cpg_sites_with_nans_sorted_df = cpg_sites_with_nans_df.sort_values(by="nan_count", ascending=False)

        return df

    def normalize_group(self, df):
        metadata_df, methyl_df = self.split_df(df)
        methyl_df = pd.DataFrame(self.scaler.fit_transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns)
        df = self.join_dfs(metadata_df, methyl_df)
        return df

    def prepare_features(self, df, is_training=True):

        ## Remove samples with more than X perc nan values across all cpg sites.
        #   This is a threshold. If this number is 0, it means
        #   filer out any samples that have any missing data. If this number is 100,
        #   it means only filter out samples that have all their site data missing.
        nan_counts = df.isna().sum(axis=1)  # Count NaNs per GSM ID
        max_nan_allowed = self.config["model.input_dim"] * (self.config["remove_nan_samples_perc_2"] / 100.0)
        gsm_ids_with_nans = nan_counts[nan_counts > max_nan_allowed].index
        df = df[~df.index.isin(gsm_ids_with_nans)]

        metadata_df, methyl_df = self.split_df(df)

        # &&&

        # # Make sure all values are numeric. Without this we were having issues with the 'mean' imputation strategy.
        # methyl_df = methyl_df.apply(pd.to_numeric, errors='coerce')

        # methyl_df = methyl_df.replace({pd.NA: np.nan, None: np.nan})
        # methyl_df = methyl_df.astype("float32")
        # methyl_df = methyl_df.fillna(np.nan)
        #
        # print("NumPy version:", np.__version__)
        # print("Pandas version:", pd.__version__)

        # import sys
        # sys.exit()

        # &&& end

        ## impute #$ docs
        if is_training:
            methyl_df = pd.DataFrame(
                self.imputer.fit_transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns
            )
        else:
            methyl_df = pd.DataFrame(
                self.imputer.transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns
            )

        ## normalize #$ docs
        if self.normalization_strategy == "per_study_per_site":
            df = self.join_dfs(metadata_df, methyl_df)
            df = df.groupby('gse_id', group_keys=False).apply(self.normalize_group)
            metadata_df, methyl_df = self.split_df(df)
        elif self.normalization_strategy == "per_site":
            if is_training:
                methyl_df = pd.DataFrame(
                    self.scaler.fit_transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns
                )
            else:
                methyl_df = pd.DataFrame(
                    self.scaler.transform(methyl_df), index=methyl_df.index, columns=methyl_df.columns
                )
        else:
            raise ValueError(f"Bad normalization_strategy: {self.normalization_strategy}")

        # # Check min and max values for each column. Needs work.
        # min_max_df = (
        #     methyl_df
        #     .agg(['min', 'max'])  # Compute min and max for each column
        #     .T  # Transpose for a tabular structure
        #     .reset_index()  # Convert index to column
        # )
        # min_max_df.columns = ['cpg_site_id', 'min_value', 'max_value']
        # min_max_df = min_max_df.sort_values(by='cpg_site_id').reset_index(drop=True)

        age_ser = metadata_df[self.age_col_str] if self.age_col_str in metadata_df.columns else None
        return methyl_df, age_ser

    @staticmethod
    def split_data_by_type(df):
        train_df = df[df["type"] == "train"]
        test_df = df[df["type"] == "verification"]
        return train_df, test_df

    def split_data_by_percent(self, df):
        return train_test_split(df, test_size=self.config["test_ratio"], random_state=42)

    def train(self, train_data, val_data):
        print(f"Training model...")
        train_dt = datetime.now()

        features_train, ages_train = self.prepare_features(train_data, is_training=True)
        features_val, ages_val = self.prepare_features(val_data, is_training=False)

        train_dataset = MethylationDataset(features_train, ages_train)
        val_dataset = MethylationDataset(features_val, ages_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.config["lr_factor"],
            patience=self.config["lr_patience"],
            threshold=self.config["lr_threshold"],  # Matches the minimum delta for improvement
        )

        # Early stopper
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        self.model = torch.compile(self.model, backend="aot_eager").to(self.device)

        for epoch in range(self.config["max_epochs"]):
            # print(f"--- Epoch '{epoch + 1}/{epochs}' --------------------")
            self.model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                # print(f"Processing batch: {i+1}/{len(train_loader)}")

                # noinspection PyTypeChecker
                features = batch["features"].to(self.device)
                # noinspection PyTypeChecker
                ages = batch["age"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features).squeeze()
                loss = self.criterions[self.config["loss_name"]](predictions, ages)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self.validate(val_loader, self.config["loss_name"])
            print(
                f"config_id '{self.config_id}', "
                f"Epoch '{epoch+1}/{self.config['max_epochs']}', "
                f"lr: '{round(lr_scheduler.get_last_lr()[0], 10)}', "
                f"Train Loss: {train_loss / len(train_loader):,.6f}, "
                f"Val Loss: {val_loss:,.6f}"
            )

            # Learning rate scheduler
            lr_scheduler.step(val_loss)

            # Early stopper
            if val_loss < best_val_loss - self.config["early_stop_threshold"]:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best state
                best_state = {
                    "model_state_dict": deepcopy(self.model.state_dict()),
                    "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
                    "predictor_state": deepcopy(self),
                }
            else:
                patience_counter += 1
            if patience_counter >= self.config["early_stop_patience"]:
                print(f"Early stopping triggered. No improvement for '{self.config['early_stop_patience']}' epochs.")
                break

        # Restore best state
        self.__dict__.update(best_state["predictor_state"].__dict__)
        self.model.load_state_dict(best_state["model_state_dict"])
        self.optimizer.load_state_dict(best_state["optimizer_state_dict"])

        result_dict = {
            loss_name: self.validate(val_loader, loss_name) for loss_name in self.criterions
        }
        latest_val_loss = result_dict[self.config["loss_name"]]
        if best_val_loss != latest_val_loss:
            raise ValueError(
                f"Missmatch between best validation loss '{best_val_loss}' "
                f"and latest validation loss '{latest_val_loss}'."
            )

        # mem.log_memory(print, "train")
        print(f"train runtime: {datetime.now() - train_dt}")

        return result_dict

    def validate(self, val_loader, loss_name):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)
                predictions = self.model(features).squeeze()
                loss = self.criterions[loss_name](predictions, ages)
                total_loss += loss.item()
        val_loss = total_loss / len(val_loader)
        return val_loss

    def cross_train(self, train_df):
        """
        Perform k-fold cross-validation on the model.

        Args:
            train_df (pd.DataFrame): The full train dataset to split into folds.

        Returns:
            dict: Average metrics across all folds.
        """

        print(f"Starting '{self.k_folds}'-fold cross-validation...")
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_counter, (train_indices, val_indices) in enumerate(kfold.split(train_df)):
            print(f"Fold '{fold_counter+1}/{self.k_folds}'...")
            train_data = train_df.iloc[train_indices]
            val_data = train_df.iloc[val_indices]

            self.__init__(self.config)
            fold_result = self.train(train_data, val_data)
            fold_metrics.append(fold_result)

        result_dict = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}

        for name, value in result_dict.items():
            print(f"Cross-validation '{name}': {value}")

        return result_dict

    def test_(self, test_df): # &&& rename after.
        print("Testing model...")

        features_test, ages_test = self.prepare_features(test_df, is_training=False)
        test_dataset = MethylationDataset(features_test, ages_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        result_dict = {}
        actual_ages_all = []
        predictions_all = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)

                predictions = self.model(features).squeeze()
                predictions_all.extend(predictions.cpu())
                actual_ages_all.extend(ages.cpu())

            predictions_all = torch.tensor(predictions_all)
            actual_ages_all = torch.tensor(actual_ages_all)

            for name, criterion in self.criterions.items():
                result_dict[name] = criterion(predictions_all, actual_ages_all).item()

        for name, value in result_dict.items():
            print(f"Test '{name}': {value}")

        return result_dict

    def predict_batch(self, new_df, ref_df):
        print("Batch predicting...")

        df = pd.concat([ref_df, new_df])

        # Add a single-value gse_id as a column, so we can easily run normalization.
        df["gse_id"] = "dummy-gse-id"

        # Preserve gsm_id for filtering and output.
        gsm_ids = new_df.index.tolist()

        features_df, _ = self.prepare_features(df, is_training=False)
        features_df = features_df[features_df.index.isin(gsm_ids)]
        dataset = MethylationDataset(features_df, None)
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)

        predictions = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                batch_predictions = self.model(features).squeeze()
                predictions.extend(batch_predictions.cpu().numpy())

        predictions_df = pd.DataFrame({
            "gsm_id": gsm_ids,
            "predicted_age_years": predictions
        }).set_index("gsm_id")

        return predictions_df


    def train_pipeline(self):

        df = self.load_data()
        if self.split_train_test_by_percent:
            train_df, test_df = self.split_data_by_percent(df)
        else:
            train_df, test_df = self.split_data_by_type(df)


        if self.k_folds == 1:
            # Regular train on a single fold.
            train_data, val_data = self.split_data_by_percent(train_df)
            result_dict = self.train(train_data, val_data)
        else:
            # k-fold train
            result_dict = self.cross_train(train_df)

        # self.save_predictor()

        # Loading a Saved Model and test again.
        # self.load()
        # Make sure that even if we shuffle test_df, we still get the same metrics.
        # test_df = test_df.sample(frac=1, random_state=24) # &&& does this even work?

        # result_dict = self.test_(test_df)

        return result_dict

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
        # prediction_df = self.predict_batch(methyl_df, ref_df)
        #
        # # Quick sanity check with ages from actual metadata.
        # metadata_df, _ = DeepMAgePredictor.split_df(df)
        # actual_age_df = metadata_df[[DeepMAgePredictor.age_col_str]]
        # predicted_age_df = prediction_df
        # predicted_age_df.index = [gsm_id[:-1] for gsm_id in predicted_age_df.index]
        #
        # prediction_df = actual_age_df.join(predicted_age_df, how="inner")
        # print(f"Predictions for new data:\n{prediction_df}")
