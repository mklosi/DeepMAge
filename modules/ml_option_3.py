import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from modules.metadata import metadata_parquet_path
from modules.ml_common import DeepMAgeBase, MethylationDataset, DataLoader
from datetime import datetime


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


class DeepMAgePredictor(DeepMAgeBase):

    model_path = "model_artifacts/model_option_3.pkl"
    metadata_df_path = "resources/metadata_derived.parquet"
    methyl_df_path = "resources/methylation_data.parquet"

    age_col_str = "actual_age_years"
    metadata_cols_of_interest = ["gse_id", "type", age_col_str] # &&& can I be reminded of the benefit of doing cls vs. self?

    def __init__(self):
        self.input_dim = 1000
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = DeepMAgeModel(input_dim=self.input_dim).to(self.device)
        self.criterion = nn.MSELoss()  # &&& use MAE. MAE aligns better with the metrics mentioned in the paper (MedAE and MAE).
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    @classmethod
    def load_model(cls, model_path):
        """Load a saved DeepMAge model."""
        instance = cls()
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        print(f"Model loaded from: {model_path}")
        return instance

    @classmethod
    def new_model(cls):
        return cls()

    def save_model(self, save_path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

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
        """&&& docs"""
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
            'nan_count': gsm_ids_with_nans.values,
            'cpg_sites_with_nan': nan_cpg_sites[gsm_ids_with_nans.index].values
        })

        ## Show which cpg site ids have missing values and the corresponding gsm_ids that contain those nans.
        nan_counts_per_cpg = df.isna().sum(axis=0)
        cpg_sites_with_nans = nan_counts_per_cpg[nan_counts_per_cpg > 0]
        nan_gsm_ids_per_cpg = df.apply(lambda col: ','.join(df.index[col.isna()]), axis=0)
        cpg_sites_with_nans_df = pd.DataFrame({
            'cpg_site_id': cpg_sites_with_nans.index,
            'nan_count': cpg_sites_with_nans.values,
            'gsm_ids_with_nan': nan_gsm_ids_per_cpg[cpg_sites_with_nans.index].values
        })




        return df

    def prepare_features(self, df, is_training=True):

        metadata_df, methyl_df = self.split_df(df)
        features = methyl_df.drop(columns=[self.age_col_str]).values




        if is_training:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)




        ages = df["age"].values if "age" in df.columns else None
        return features, ages

    @staticmethod
    def split_data(df):
        train_df = df[df["type"] == "train"]
        test_df = df[df["type"] == "verification"]
        return train_df, test_df

    def train(self, train_df):




        self.prepare_features(train_df, is_training=True)


        train_dataset = MethylationDataset(train_df)



        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features).squeeze()
                loss = self.criterion(predictions, ages)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}")

    def validate(self, val_loader):
        """Evaluate the model on a test/validation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)
                predictions = self.model(features).squeeze()
                loss = self.criterion(predictions, ages)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def test(self, test_loader):
        pass

    def predict_batch(self, features):
        """Predict ages for a batch of features."""
        self.model.eval()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy()
        return predictions

    @classmethod
    def train_pipeline(cls):

        predictor = cls.new_model()

        df = predictor.load_data()

        # Regular train on a single fold, test, and save a model.
        train_df, test_df = predictor.split_data(df)
        predictor.train(train_df)
        predictor.test(test_df)
        predictor.save_model(cls.model_path)

        # Loading a Saved Model and test again.
        predictor = cls.load_model(cls.model_path)
        predictor.test(test_df)

        # --------------------





        val_dataset = MethylationDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Train the model
        predictor.train(train_loader, val_loader, epochs=50)

        # Save the model
        predictor.save_model(predictor.model_path)




if __name__ == "__main__":
    DeepMAgePredictor.train_pipeline()
