import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, input_dim):
        super().__init__()
        self.model = DeepMAgeModel(input_dim=input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    @classmethod
    def load_model(cls, model_path, input_dim):
        """Load a saved DeepMAge model."""
        instance = cls(input_dim)
        instance.model.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.model.eval()
        print(f"Model loaded from: {model_path}")
        return instance

    def train(self, train_loader, val_loader, epochs=50):
        """Train the DeepMAge model."""
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

            val_loss = self.test(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss / len(train_loader)}, Validation Loss: {val_loss}")

    def test(self, data_loader):
        """Evaluate the model on a test/validation set."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                features = batch["features"].to(self.device)
                ages = batch["age"].to(self.device)
                predictions = self.model(features).squeeze()
                loss = self.criterion(predictions, ages)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def predict_batch(self, features):
        """Predict ages for a batch of features."""
        self.model.eval()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(features).cpu().numpy()
        return predictions


if __name__ == "__main__":
    # Initialize and train the DeepMAgePredictor
    input_dim = 1000  # Number of CpG sites
    predictor = DeepMAgePredictor(input_dim=input_dim)

    # Load data
    data_path = "resources/methylation_data.parquet"
    df = predictor.load_data(data_path)

    # Prepare features and split
    features, ages = predictor.prepare_features(df, is_training=True)
    X_train, y_train, X_val, y_val = predictor.split_data(features, ages)

    # Create DataLoaders
    train_dataset = MethylationDataset(X_train, y_train)
    val_dataset = MethylationDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train the model
    predictor.train(train_loader, val_loader, epochs=50)

    # Save the model
    predictor.save_model(predictor.model_path)
