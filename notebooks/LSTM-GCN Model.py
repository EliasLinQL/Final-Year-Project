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
import torch.nn.functional as F
import torch_optimizer as optim_lookahead

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
def prepare_data(features_df, sequence_length=30):
    """
    Prepare time-series data using return_lag_1 to return_lag_6 as input features.

    Args:
        features_df (DataFrame): The feature dataframe containing return_lag_1 to return_lag_6.
        sequence_length (int): The length of the time-series window.

    Returns:
        X (numpy array): Time-series input features.
        y (numpy array): Corresponding target values.
    """
    # Select return_lag_1 to return_lag_6 as input features
    lag_columns = [f'BTCUSDT_return_lag_{i}' for i in range(1, 7)]
    data = features_df[lag_columns].values  # Keep only lag features

    # Use return_lag_6 as the target variable (same as before)
    target = features_df['BTCUSDT_return_lag_1'].values

    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])  # Take past 30 timesteps of features
        y.append(target[i])  # Predict return_lag_6

    return np.array(X), np.array(y)


# Set parameters
sequence_length = 30  # Keep the same sequence length
X, y = prepare_data(features_df, sequence_length=sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load data using DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ----------------- Optimized LSTM Model (BiLSTM + Self-Attention + GAP + Residual) -----------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Self-Attention for feature importance
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, batch_first=True)

        # Residual Connection
        self.residual_fc = nn.Linear(input_dim, hidden_dim * 2)  # Project input to match LSTM output dim

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = self.residual_fc(x)  # Project input to match LSTM output

        lstm_out, _ = self.lstm(x)  # Output shape: [batch, seq_len, hidden_dim * 2]
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # Self-Attention

        # Apply Global Average Pooling (GAP) to both LSTM and Residual identity
        attn_output = torch.mean(attn_output, dim=1)  # Shape: [batch, hidden_dim * 2]
        identity = torch.mean(identity, dim=1)  # Ensure identity has same shape as attn_output

        # Apply residual connection and layer normalization
        attn_output = self.layer_norm(attn_output + identity)

        # Final fully connected output
        output = self.fc(self.dropout(attn_output))  # Dropout before final FC layer
        return output


# ----------------- Optimized GCN Model (Deeper GCN + Improved Residual) -----------------
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Better gradient flow

        # Improved Residual Connection
        self.residual_fc = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index):
        identity = self.residual_fc(x)  # Residual connection projection

        x = self.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = self.conv4(x, edge_index)

        x = x + identity  # Add Residual Connection
        x = self.layer_norm(self.bn4(x))  # Normalize final output
        return x

# ----------------- Optimized LSTM-GCN Combined Model -----------------
class LSTMGCNModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_num_layers,
                 gcn_input_dim, gcn_hidden_dim, gcn_output_dim,
                 combined_output_dim, dropout=0.3):
        super(LSTMGCNModel, self).__init__()

        # LSTM Model
        self.lstm = LSTMModel(lstm_input_dim, lstm_hidden_dim, lstm_num_layers, combined_output_dim, dropout)

        # GCN Model
        self.gcn = GCNModel(gcn_input_dim, gcn_hidden_dim, gcn_output_dim, dropout)
        self.gcn_fc = nn.Linear(gcn_output_dim, combined_output_dim)

        # Feature Interaction Layer (FIL)
        self.feature_interaction = nn.Linear(2 * combined_output_dim, 2 * combined_output_dim)

        # Improved Multi-Head Attention for feature fusion
        self.attention = nn.MultiheadAttention(embed_dim=2 * combined_output_dim, num_heads=4, batch_first=True)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(2 * combined_output_dim)

        # Dropout Regularization
        self.dropout = nn.Dropout(dropout)

        # Final Fully Connected Layer
        self.final_fc = nn.Linear(2 * combined_output_dim, 1)

    def forward(self, lstm_x, gcn_x, edge_index):
        # Process LSTM input
        lstm_out = self.lstm(lstm_x)  # Shape: [batch, combined_output_dim]

        # Process GCN input
        gcn_out = self.gcn(gcn_x, edge_index)  # Shape: [num_nodes, gcn_output_dim]
        gcn_out = self.gcn_fc(gcn_out)  # Map GCN output to LSTM dimension

        # Compute adaptive mean over GCN output and match batch size
        gcn_out = torch.mean(gcn_out, dim=0, keepdim=True).repeat(lstm_out.size(0), 1)

        # Concatenate LSTM and GCN outputs
        combined = torch.cat([lstm_out, gcn_out], dim=-1)

        # Feature Interaction Layer
        combined = self.feature_interaction(combined)

        # Apply Multi-Head Attention for better feature fusion
        attn_input = combined.unsqueeze(1)  # [batch, 1, 2 * combined_output_dim]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(1)

        # Apply Layer Normalization and Dropout
        combined = self.layer_norm(attn_output)
        combined = self.dropout(combined)

        # Output prediction
        output = self.final_fc(combined)
        return output


# ----------------- Model Initialization -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMGCNModel(
    lstm_input_dim=X.shape[2],
    lstm_hidden_dim=512,
    lstm_num_layers=4,
    gcn_input_dim=features_df.shape[1] - 1,
    gcn_hidden_dim=32,
    gcn_output_dim=8,
    combined_output_dim=16,
    dropout=0.3
).to(device)

# Apply Xavier Uniform Initialization to Model Weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

print(model)

# ----------------- Training Configuration -----------------
criterion = nn.SmoothL1Loss()  # Huber Loss (less sensitive to outliers)
base_optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Lookahead Wrapper for AdamW (improves stability)
class LookaheadOptimizer(optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5):
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.counter = 0
        self.slow_params = [p.clone().detach() for p in self.param_groups[0]['params']]

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.counter += 1
        if self.counter >= self.k:
            for slow, fast in zip(self.slow_params, self.param_groups[0]['params']):
                slow += self.alpha * (fast - slow)
                fast.copy_(slow)
            self.counter = 0
        return loss

# Wrap AdamW with Lookahead optimizer
optimizer = optim_lookahead.Lookahead(base_optimizer, k=5, alpha=0.5)  # k: synchronization steps, alpha: slow weights update

# Use CosineAnnealingWarmRestarts for better learning rate scheduling
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.optimizer, T_0=10, T_mult=2)


# ----------------- Optimized Model Training (Enhanced with Multiple Models) -----------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, cooldown=2):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.cooldown_counter = 0
        self.cooldown = cooldown

    def check_early_stop(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.cooldown_counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.cooldown_counter += 1
                if self.cooldown_counter >= self.cooldown:
                    return True
        return False


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def update_running_loss(running_loss, new_loss, beta=0.9):
    return beta * running_loss + (1 - beta) * new_loss if running_loss else new_loss

# Training Loop with Multiple Models
num_models = 3
is_first_model = True
for model_index in range(num_models):
    # Reinitialize early stopping for each model
    early_stopper = EarlyStopping(patience=5, cooldown=2)

    # Check if it is the first model, skip reinitialization if so
    if not is_first_model:
        # Reinitialize model, optimizer, and scheduler for each run
        model = LSTMGCNModel(
            lstm_input_dim=X.shape[2],
            lstm_hidden_dim=512,
            lstm_num_layers=4,
            gcn_input_dim=features_df.shape[1] - 1,
            gcn_hidden_dim=32,
            gcn_output_dim=8,
            combined_output_dim=16,
            dropout=0.3
        ).to(device)
        # Reapply weight initialization to ensure randomness
        model.apply(initialize_weights)

        optimizer = optim_lookahead.Lookahead(
            optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5), k=5, alpha=0.5
        )
        # Clear optimizer state to remove any residual effects
        optimizer.zero_grad(set_to_none=True)

        # Reinitialize the scheduler to ensure it starts from scratch
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.optimizer, T_0=10, T_mult=2)

    # After first model, set flag to False
    is_first_model = False

    running_loss = None
    num_epochs = 75
    model_losses = []  # List to store loss for each epoch
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            gcn_input = torch.rand(correlation_matrix.shape[0], features_df.shape[1] - 1).to(device)
            X_batch, y_batch = mixup_data(X_batch, y_batch)
            optimizer.zero_grad()
            outputs = model(X_batch, gcn_input, edge_index.to(device))
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        model_losses.append(avg_loss)  # Store average loss for each epoch
        # Reset scheduler step to prevent continuation of the previous model's state
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer.optimizer, T_0=10, T_mult=2)
        scheduler.step()
        running_loss = update_running_loss(running_loss, avg_loss)
        print(f"Model {model_index+1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f} (EMA: {running_loss:.4f})")
        if early_stopper.check_early_stop(avg_loss):
            print("Early stopping triggered!")
            break
    model_save_path = os.path.join(project_root, "models", f"model_{model_index+1}.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model {model_index+1} saved at {model_save_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(model_losses, label=f'Model {model_index+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model {model_index+1} Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(project_root, 'results', f'model_{model_index+1}_loss.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved at {loss_plot_path}")
    plt.show()

# ----------------- Model Testing (Enhanced for Multiple Models) -----------------
for model_index in range(num_models):
    model_path = os.path.join(project_root, "models", f"model_{model_index+1}.pt")
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT']
    for symbol in symbols:
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
        data_file = os.path.join(data_dir, f"{symbol}_2021_2024.csv")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['open_time'])
        df.set_index('date', inplace=True)
        prices = df['close'][-len(y_test):].values
        min_len = min(len(prices[:-1]), len(y_true))
        true_prices = prices[-min_len-1:-1] * (1 + y_true[-min_len:])
        predicted_prices = prices[-min_len-1:-1] * (1 + y_pred[-min_len:])
        test_index = df.index[-min_len:]
        comparison_df = pd.DataFrame({'Actual Price': true_prices, 'Predicted Price': predicted_prices}, index=test_index)
        output_dir = os.path.join(project_root, 'results', f'model_{model_index+1}')
        os.makedirs(output_dir, exist_ok=True)
        output_plot_path = os.path.join(output_dir, f'{symbol}_Actual_vs_Predicted.png')
        comparison_df.to_csv(os.path.join(output_dir, f'{symbol}_Predictions.csv'), index_label='Date')
        plt.figure(figsize=(14, 7))
        plt.plot(comparison_df.index, comparison_df['Actual Price'], label=f'Actual {symbol} Price', color='blue')
        plt.plot(comparison_df.index, comparison_df['Predicted Price'], label=f'Predicted {symbol} Price', color='orange', linestyle='dashed')
        plt.title(f'{symbol} Actual vs Predicted Prices (Model {model_index+1})')
        plt.xlabel('Date')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_plot_path)
        print(f'Plot saved at {output_plot_path}')
        plt.show()
