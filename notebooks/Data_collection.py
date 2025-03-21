# backend/data_collection.py

import os
import time
import pandas as pd
import requests

# Binance API base URL
BASE_URL = 'https://api.binance.com'

# 项目根目录和数据目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

def fetch_crypto_data(symbols, start_time, end_time):
    limit = 1000
    file_paths = []

    start_date_str = time.strftime('%Y-%m-%d', time.gmtime(start_time / 1000))
    end_date_str = time.strftime('%Y-%m-%d', time.gmtime(end_time / 1000))

    for symbol in symbols:
        output_file = os.path.join(data_dir, f'{symbol}_{start_date_str}_{end_date_str}.csv')
        final_df = pd.DataFrame()
        current_start_time = start_time

        print(f"🔍 Fetching data for {symbol} from {start_date_str} to {end_date_str}")

        try:
            while current_start_time < end_time:
                url = f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval=4h&limit={limit}&startTime={current_start_time}&endTime={end_time}"
                response = requests.get(url)
                if response.status_code != 200:
                    print(f"⚠️ Failed to fetch data for {symbol}. Status: {response.status_code}")
                    break

                data = response.json()
                if not data or isinstance(data, dict):
                    print(f"⚠️ No data returned or error for {symbol}")
                    break

                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

                final_df = pd.concat([final_df, df], ignore_index=True)

                current_start_time = int(df['open_time'].iloc[-1].timestamp() * 1000) + 1

            if not final_df.empty:
                final_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # ✅ 使用utf-8-sig防止编码问题
                file_paths.append(output_file)
                print(f"✅ Saved: {output_file}")
            else:
                print(f"⚠️ No data collected for {symbol}")

        except Exception as e:
            print(f"❌ Exception occurred while processing {symbol}: {e}")

    return file_paths
