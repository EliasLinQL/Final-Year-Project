import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set project directory
project_root = r"D:\Y3\FYP\Final-Year-Project\Final-Year-Project"
result_dir = os.path.join(project_root, "results")

# Parameters
FEE = 0.0002  # Transaction fee (0.02%)
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT']
num_models = 3

for model_index in range(num_models):
    # Initialize metrics dictionary
    metrics = {
        'Symbol': [],
        'Win Rate': [],
        'Total Return': [],
        'Max Drawdown': [],
        'Sharpe Ratio': [],
        'Total Trades': []
    }

    for symbol in symbols:
        # Load market data
        data_file = os.path.join(project_root, "data", f"{symbol}_2021_2024.csv")
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['open_time'])
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Load model predictions
        predictions_file = os.path.join(result_dir, f"model_{model_index+1}", f"{symbol}_Predictions.csv")
        predictions = pd.read_csv(predictions_file)
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        predictions.set_index('Date', inplace=True)

        # Align prediction data with market data timeframe
        aligned_index = predictions.index.intersection(df.index)
        df = df.loc[aligned_index]
        predictions = predictions.loc[aligned_index]

        # Convert predicted prices to returns
        y_pred = predictions['Predicted Price'].values / df['close'].values - 1

        # Ensure data length consistency
        prices = df['close'].values
        actual_returns = df['close'].pct_change().values[1:]
        y_pred = y_pred[:len(prices) - 1]

        # Adjust length
        df = df.iloc[:len(y_pred)]
        df['y_pred'] = y_pred

        # Define positions: Long (1), Short (-1), Neutral (0)
        df['position'] = np.where(df['y_pred'] > FEE, 1, np.where(df['y_pred'] < -FEE, -1, 0))
        df['trades'] = abs(df['position'].diff().fillna(0))

        # Compute strategy returns (including transaction costs)
        df['return'] = df['close'].pct_change()
        df['strategy_return'] = (df['position'].shift(1) * df['return']) - (FEE * df['trades'])
        df['cumulative_market'] = (1 + df['return']).cumprod()
        df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

        # Compute backtest metrics
        actual_directions = np.sign(df['return'].iloc[:len(y_pred)])
        predicted_directions = np.sign(y_pred)
        correct_predictions = (actual_directions == predicted_directions).sum()
        total_predictions = len(predicted_directions)
        win_rate = correct_predictions / total_predictions
        max_drawdown = (df['cumulative_strategy'].cummax() - df['cumulative_strategy']).max()
        sharpe_ratio = (df['strategy_return'].mean() / df['strategy_return'].std()) * (252 ** 0.5)

        # Print metrics
        print(f"Model {model_index+1}, {symbol} - Win Rate: {win_rate:.2f}, Total Return: {df['cumulative_strategy'].iloc[-1]:.2f}, Max Drawdown: {max_drawdown:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")

        # Store metrics in dictionary
        metrics['Symbol'].append(symbol)
        metrics['Win Rate'].append(win_rate)
        metrics['Total Return'].append(df['cumulative_strategy'].iloc[-1])
        metrics['Max Drawdown'].append(max_drawdown)
        metrics['Sharpe Ratio'].append(sharpe_ratio)
        metrics['Total Trades'].append(df['trades'].sum())

        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['cumulative_market'], label="Market (HODL)", color='blue')
        plt.plot(df.index, df['cumulative_strategy'], label="Strategy", color='orange')
        plt.title(f"Backtest Performance - {symbol} (Model {model_index+1})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.grid(True)
        backtest_plot_path = os.path.join(result_dir, f"model_{model_index+1}", f"{symbol}_Backtest.png")
        plt.savefig(backtest_plot_path)
        print(f"Backtest plot saved at {backtest_plot_path}")
        plt.close()

    # Save all metrics in a single CSV file for each model
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(result_dir, f"model_{model_index+1}_Backtest_Metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved at {metrics_path}")
