import os
import time
import pandas as pd
import requests

# Base URL for Binance API
BASE_URL = 'https://api.binance.com'

# Time settings: From January 1, 2021, to November 1, 2024 (timestamps in milliseconds)
start_time = int(time.mktime(time.strptime('2021-01-01', '%Y-%m-%d')) * 1000)
end_time = int(time.mktime(time.strptime('2025-01-01', '%Y-%m-%d')) * 1000)

# Maximum number of records per request
limit = 1000

# Get the current script directory (notebooks folder)
notebooks_dir = os.path.dirname(os.path.abspath(__file__))

# Project root directory (parent directory of notebooks)
project_root = os.path.abspath(os.path.join(notebooks_dir, '..'))

# Path to the data folder
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# List of trading pairs
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT']

# Loop through each trading pair to fetch data
for symbol in symbols:
    output_file = os.path.join(data_dir, f'{symbol}_2021_2024.csv')

    # Initialize DataFrame
    final_df = pd.DataFrame()

    # Loop to fetch data
    current_start_time = start_time
    while current_start_time < end_time:
        url = (
                BASE_URL +
                f'/api/v3/klines?symbol={symbol}&interval=4h&limit={limit}&startTime={current_start_time}&endTime={end_time}'
        )
        print(f"Fetching data for {symbol} from: {url}")
        response = requests.get(url)
        data = response.json()

        if len(data) == 0:
            break  # Exit the loop if no data is returned

        # Convert the data into a DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert timestamps to date
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Select necessary columns
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

        # Append the data to the final DataFrame
        final_df = pd.concat([final_df, df], ignore_index=True)

        # Update start_time to the last recorded open_time
        current_start_time = int(df['open_time'].iloc[-1].timestamp() * 1000) + 1

    # Save the data to a CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Data for {symbol} successfully saved to {output_file}")

    # Print data preview
    print(final_df.head())
