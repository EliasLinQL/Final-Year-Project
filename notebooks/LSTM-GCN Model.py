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

# ----------------- Data Preparation -----------------

# Set the project root and data directory
project_root = r"D:\Y3\FYP\Final-Year-Project\Final-Year-Project"
data_dir = os.path.join(project_root, "data")

# Load feature data
features_file = os.path.join(data_dir, "features.csv")
features_df = pd.read_csv(features_file, index_col=0)
print("Features loaded successfully.")

# Load correlation matrix
correlation_matrix_file = os.path.join(data_dir, "correlation_matrix.csv")
correlation_matrix = pd.read_csv(correlation_matrix_file, index_col=0)
print("Correlation matrix loaded successfully.")

# Load graph edge list
graph_edges_file = os.path.join(data_dir, "graph_edges.csv")
edges_df = pd.read_csv(graph_edges_file)
print("Graph edges loaded successfully.")

# Map node names to integer indices
unique_nodes = pd.concat([edges_df['source'], edges_df['target']]).unique()
node_mapping = {name: idx for idx, name in enumerate(unique_nodes)}

# Convert source and target columns to indices
edges_df['source'] = edges_df['source'].map(node_mapping)
edges_df['target'] = edges_df['target'].map(node_mapping)

# Convert edge list to PyTorch Geometric format
edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long)
print(f"Edge index shape: {edge_index.shape}")

# Define a PyTorch Dataset for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Prepare time series data
def prepare_data(features_df, target_column, sequence_length=30):
    data = features_df.drop(columns=[target_column]).values
    target = features_df[target_column].values

    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(target[i])

    return np.array(X), np.array(y)

# Parameters
sequence_length = 30
target_column = 'BTCUSDT_return_lag_6'

# Split data into training and testing sets
X, y = prepare_data(features_df, target_column, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load data using DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------- Improved LSTM Model (BiLSTM) -----------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Adjusted for BiLSTM output

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenating forward and backward hidden states
        output = self.fc(hidden)
        return output

# ----------------- Improved GCN Model -----------------
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
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
        x = self.conv3(x, edge_index)
        return x

# ----------------- Improved LSTM-GCN Combined Model -----------------
class LSTMGCNModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layers,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 combined_output_dim, dropout=0.3):
        super(LSTMGCNModel, self).__init__()
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
        output = self.final_fc(combined)
        return self.sigmoid(output)

# ----------------- Model Initialization -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMGCNModel(
    lstm_input_dim=X.shape[2],
    lstm_hidden_dim=64,
    lstm_num_layers=2,
    gcn_input_dim=features_df.shape[1] - 1,
    gcn_hidden_dim=16,
    gcn_output_dim=8,
    combined_output_dim=16
).to(device)
print(model)

# ----------------- Training Configuration -----------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

# Define Cyclical Learning Rate scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.005,  # Maximum learning rate
    steps_per_epoch=len(train_loader),
    epochs=num_epochs
)

# ----------------- Model Training -----------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        Initialize EarlyStopping with:
        - patience: Number of epochs to wait after no improvement.
        - min_delta: Minimum change in loss to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def check_early_stop(self, val_loss):
        """
        Check if training should be stopped early.
        - val_loss: Current validation loss.
        """
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if loss improves
        else:
            self.counter += 1  # Increment counter if no improvement

        return self.counter >= self.patience


# Initialize EarlyStopping
early_stopper = EarlyStopping(patience=5)  # Stop training if no improvement for 5 epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        gcn_input = torch.rand(correlation_matrix.shape[0], features_df.shape[1] - 1).to(device)

        optimizer.zero_grad()
        outputs = model(X_batch, gcn_input, edge_index.to(device))
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

        scheduler.step()  # Update learning rate
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()}")

    # Check Early Stopping
    if early_stopper.check_early_stop(avg_loss):
        print("Early stopping triggered!")
        break

# ----------------- Model Testing -----------------
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        gcn_input = torch.rand(correlation_matrix.shape[0], features_df.shape[1] - 1).to(device)
        outputs = model(X_batch, gcn_input, edge_index.to(device))
        y_pred.extend(outputs.squeeze().cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Load BTCUSDT trading data
btc_data_file = os.path.join(data_dir, "BTCUSDT_2021_2024.csv")
btc_df = pd.read_csv(btc_data_file)

btc_df['date'] = pd.to_datetime(btc_df['open_time'])
btc_df.set_index('date', inplace=True)
btc_prices = btc_df['close'][-len(y_test):].values

# Ensure lengths match
if len(y_true) > len(btc_prices[:-1]):
    y_true = y_true[:len(btc_prices[:-1])]
elif len(y_true) < len(btc_prices[:-1]):
    btc_prices = btc_prices[:len(y_true) + 1]

true_prices = btc_prices[:-1] * (1 + y_true)
predicted_prices = btc_prices[:-1] * (1 + y_pred[:len(y_true)])

# Plot actual vs. predicted
test_index = btc_df.index[-len(true_prices):]
comparison_df = pd.DataFrame({'Actual Price': true_prices, 'Predicted Price': predicted_prices}, index=test_index)

plt.figure(figsize=(14, 7))
plt.plot(comparison_df.index, comparison_df['Actual Price'], label='Actual BTCUSDT Price', color='blue', linewidth=2)
plt.plot(comparison_df.index, comparison_df['Predicted Price'], label='Predicted BTCUSDT Price', color='orange', linestyle='dashed', linewidth=2)
plt.title('BTCUSDT: Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.legend()
plt.grid(True)

# Save the plot
output_plot_path = os.path.join(project_root, "results", "BTCUSDT_Actual_vs_Predicted.png")
plt.savefig(output_plot_path, format='png', dpi=300)
print(f"Comparison chart saved to {output_plot_path}")

# Save prediction results to CSV
output_csv_path = os.path.join(project_root, "results", "BTCUSDT_Predictions.csv")
comparison_df.to_csv(output_csv_path, index_label='Date')
print(f"Predicted results saved to {output_csv_path}")

plt.tight_layout()
plt.show()