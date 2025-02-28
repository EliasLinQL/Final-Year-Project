import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set project directory
project_root = r"D:\Y3\FYP\Final-Year-Project\Final-Year-Project"
result_dir = os.path.join(project_root, "results")

# Load market data
btc_df = pd.read_csv(os.path.join(project_root, "data", "BTCUSDT_2021_2024.csv"))
btc_df['date'] = pd.to_datetime(btc_df['open_time'])  # Convert time column to datetime format
btc_df.set_index('date', inplace=True)  # Set date as index
btc_df = btc_df[['open', 'high', 'low', 'close', 'volume']]  # Keep only the necessary columns

# Load model predictions
predictions = pd.read_csv(os.path.join(result_dir, "BTCUSDT_Predictions.csv"))
predictions['Date'] = pd.to_datetime(predictions['Date'])  # Correct column name and convert to datetime format
predictions.set_index('Date', inplace=True)  # Set date as index

# Resample predictions to every 4 hours
# Use the resample method to select the first prediction every 4 hours
four_hour_predictions = predictions.resample('4h').first()

# Align prediction data with market data timeframe
aligned_index = four_hour_predictions.index.intersection(btc_df.index)  # Ensure index consistency
btc_df = btc_df.loc[aligned_index]
four_hour_predictions = four_hour_predictions.loc[aligned_index]

# Convert predicted prices to returns
y_pred = four_hour_predictions['Predicted Price'].values / btc_df['close'].values - 1

# Ensure data length consistency
btc_prices = btc_df['close'].values
actual_returns = btc_df['close'].pct_change().values[1:]  # Remove the first NaN value
y_pred = y_pred[:len(btc_prices) - 1]  # Ensure length matches btc_prices[:-1]

# **New condition: Only go long when returns exceed transaction fees; remain neutral otherwise**
FEE = 0.00  # Transaction fee (0.1%)
btc_df = btc_df.iloc[:len(y_pred)]  # Adjust btc_df length
btc_df['y_pred'] = y_pred  # Add predicted returns
btc_df['position'] = np.where(
    btc_df['y_pred'] > FEE, 1,  # Go long if predicted return exceeds the fee
    0  # Otherwise, stay neutral
)

# Count the number of trades
btc_df['trades'] = abs(btc_df['position'].diff().fillna(0))  # Compute trade occurrences

# Compute strategy returns (including transaction costs)
btc_df['return'] = btc_df['close'].pct_change()  # Market returns
btc_df['strategy_return'] = (
    btc_df['position'].shift(1) * btc_df['return'] - FEE * btc_df['trades']
)  # Strategy returns

# Compute cumulative returns
btc_df['cumulative_market'] = (1 + btc_df['return']).cumprod()
btc_df['cumulative_strategy'] = (1 + btc_df['strategy_return']).cumprod()

# Compute prediction accuracy
actual_directions = np.sign(btc_df['return'].iloc[:len(y_pred)].values)  # Market actual directions
predicted_directions = np.sign(y_pred)  # Predicted directions
correct_predictions = (actual_directions == predicted_directions).sum()
total_predictions = len(predicted_directions)
win_rate = correct_predictions / total_predictions

# Compute backtest metrics
max_drawdown = (btc_df['cumulative_strategy'].cummax() - btc_df['cumulative_strategy']).max()
sharpe_ratio = (btc_df['strategy_return'].mean() / btc_df['strategy_return'].std()) * (252 ** 0.5)

# Print results
print(f"Prediction Win Rate: {win_rate:.2f}")
print(f"Total Return: {btc_df['cumulative_strategy'].iloc[-1]:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Trades: {btc_df['trades'].sum()}\n")  # Print trade count

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(btc_df.index, btc_df['cumulative_market'], label="Market (HODL)", color='blue')
plt.plot(btc_df.index, btc_df['cumulative_strategy'], label="Strategy (4H Trades, No Shorting)", color='orange')
plt.legend()
plt.title("Backtest Performance: Every 4 Hours Decision Making with Transaction Costs, No Shorting")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.grid(True)
plt.savefig(os.path.join(result_dir, "Backtest_Performance_4H_No_Shorting.png"), dpi=300)
plt.show()
