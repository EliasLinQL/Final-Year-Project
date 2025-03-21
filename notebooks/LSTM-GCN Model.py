# model_training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

def train_model(project_root=r"C:\Users\32561\Desktop\lqf"):
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    features_df = pd.read_csv(os.path.join(data_dir, "features.csv"), index_col=0)
    correlation_matrix = pd.read_csv(os.path.join(data_dir, "correlation_matrix.csv"), index_col=0)
    edges_df = pd.read_csv(os.path.join(data_dir, "graph_edges.csv"))

    unique_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
    node_mapping = {name: idx for idx, name in enumerate(unique_nodes)}
    edges_df['source'] = edges_df['source'].map(node_mapping)
    edges_df['target'] = edges_df['target'].map(node_mapping)
    edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def prepare_data(features_df, target_column, sequence_length=30):
        data = features_df.drop(columns=[target_column]).values
        target = features_df[target_column].values
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    sequence_length = 30
    target_column = 'BTCUSDT_return_lag_6'
    X, y = prepare_data(features_df, target_column, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=32, shuffle=False)

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.fc(hidden)

    class GCNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
        def forward(self, x, edge_index):
            x = self.relu(self.bn1(self.conv1(x, edge_index)))
            x = self.dropout(x)
            x = self.relu(self.bn2(self.conv2(x, edge_index)))
            x = self.dropout(x)
            return self.conv3(x, edge_index)

    class LSTMGCNModel(nn.Module):
        def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layers,
                     gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                     combined_output_dim, dropout=0.3):
            super().__init__()
            self.lstm = LSTMModel(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, combined_output_dim, dropout)
            self.gcn = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim)
            self.gcn_fc = nn.Linear(gcn_output_dim, combined_output_dim)
            self.final_fc = nn.Linear(2 * combined_output_dim, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, lstm_x, gcn_x, edge_index):
            lstm_out = self.lstm(lstm_x)
            gcn_out = self.gcn(gcn_x, edge_index)
            gcn_out = self.gcn_fc(gcn_out)
            gcn_out = torch.mean(gcn_out, dim=0, keepdim=True).repeat(lstm_out.size(0), 1)
            combined = torch.cat([lstm_out, gcn_out], dim=-1)
            return self.sigmoid(self.final_fc(combined))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMGCNModel(
        lstm_input_dim=X.shape[2], lstm_hidden_dim=64, lstm_num_layers=2,
        gcn_input_dim=features_df.shape[1] - 1, gcn_hidden_dim=16, gcn_output_dim=8,
        combined_output_dim=16
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=50)

    class EarlyStopping:
        def __init__(self, patience=5, min_delta=1e-4):
            self.patience, self.min_delta = patience, min_delta
            self.best_loss, self.counter = float("inf"), 0
        def check(self, loss):
            if loss + self.min_delta < self.best_loss:
                self.best_loss, self.counter = loss, 0
            else:
                self.counter += 1
            return self.counter >= self.patience

    stopper = EarlyStopping()

    for epoch in range(50):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            gcn_input = torch.rand(correlation_matrix.shape[0], features_df.shape[1] - 1).to(device)
            optimizer.zero_grad()
            output = model(Xb, gcn_input, edge_index.to(device))
            loss = criterion(output.squeeze(), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        if stopper.check(avg_loss):
            print("Early stopping triggered.")
            break

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            gcn_input = torch.rand(correlation_matrix.shape[0], features_df.shape[1] - 1).to(device)
            output = model(Xb, gcn_input, edge_index.to(device))
            y_pred.extend(output.squeeze().cpu().numpy())
            y_true.extend(yb.cpu().numpy())

    btc_file = next((f for f in os.listdir(data_dir) if "BTCUSDT" in f and f.endswith('.csv')), None)
    if btc_file:
        btc_data = pd.read_csv(os.path.join(data_dir, btc_file))
        btc_data['date'] = pd.to_datetime(btc_data['open_time'])
        btc_data.set_index('date', inplace=True)
        btc_prices = btc_data['close'][-len(y_true)-1:].values
        true_prices = btc_prices[:-1] * (1 + np.array(y_true))
        pred_prices = btc_prices[:-1] * (1 + np.array(y_pred[:len(y_true)]))
        index = btc_data.index[-len(true_prices):]
        df = pd.DataFrame({'Actual Price': true_prices, 'Predicted Price': pred_prices}, index=index)
        df.to_csv(os.path.join(results_dir, "BTCUSDT_Predictions.csv"))

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Actual Price'], label='Actual')
        plt.plot(df.index, df['Predicted Price'], label='Predicted', linestyle='dashed')
        plt.legend()
        plt.title("BTCUSDT Price Prediction")
        plt.savefig(os.path.join(results_dir, "BTCUSDT_Actual_vs_Predicted.png"), dpi=300)
        plt.tight_layout()
        plt.close()
        print("✅ Training complete and results saved.")
    else:
        print("⚠️ BTCUSDT csv not found, skipping price visualization.")

    return {
        "status": "success",
        "message": "Model training completed.",
        "results_path": os.path.join(results_dir, "BTCUSDT_Predictions.csv")
    }

# Optional standalone execution
if __name__ == "__main__":
    result = train_model()
    print(result)
