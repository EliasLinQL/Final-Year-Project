import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import pandas as pd
import requests
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 允许跨域请求

# Binance API Base URL
BASE_URL = 'https://api.binance.com'

# 数据存储目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# ----------------- 接口：数据采集 + 处理 -----------------
@app.route('/api/fetch_crypto_data', methods=['POST'])
def fetch_crypto_data():
    try:
        request_data = request.json
        print("✅ Received Request Data:", request_data)

        start_date = request_data.get("start_date", "2021-01-01")
        end_date = request_data.get("end_date", "2025-01-01")
        symbols = request_data.get("symbols", ["BTCUSDT", "ETHUSDT"])

        start_time = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')) * 1000)
        end_time = int(time.mktime(time.strptime(end_date, '%Y-%m-%d')) * 1000)

        print(f"📅 Fetching data from {start_date} to {end_date} for symbols: {symbols}")

        file_paths = fetch_and_save_crypto_data(symbols, start_time, end_time)

        if file_paths:
            print("🚀 Fetching complete! Now executing Data_processing.py...")
            processing_result = execute_data_processing()
        else:
            processing_result = {"status": "error", "message": "No data fetched"}

        return jsonify({
            "message": "Data fetch request processed",
            "processing_status": processing_result["status"],
            "processing_message": processing_result["message"]
        }), 200

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ----------------- 数据采集函数 -----------------
def fetch_and_save_crypto_data(symbols, start_time, end_time):
    limit = 1000
    file_paths = []

    start_date_str = time.strftime('%Y-%m-%d', time.gmtime(start_time / 1000))
    end_date_str = time.strftime('%Y-%m-%d', time.gmtime(end_time / 1000))

    for symbol in symbols:
        output_file = os.path.join(data_dir, f'{symbol}_{start_date_str}_{end_date_str}.csv')
        final_df = pd.DataFrame()
        current_start_time = start_time

        while current_start_time < end_time:
            url = f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval=4h&limit={limit}&startTime={current_start_time}&endTime={end_time}"
            print(f"🔍 Fetching data for {symbol} from: {url}")
            response = requests.get(url)
            data = response.json()

            if not data or isinstance(data, dict):
                print(f"⚠️ No data returned for {symbol}")
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
            final_df.to_csv(output_file, index=False)
            file_paths.append(output_file)
            print(f"✅ Data for {symbol} saved to {output_file}")

    return file_paths

# ----------------- 数据处理脚本执行函数 -----------------
def execute_data_processing():
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'Data_processing.py')
        python_executable = sys.executable

        if os.path.exists(script_path):
            print(f"⚙️ Executing {script_path} ...")
            result = subprocess.run([python_executable, script_path], capture_output=True, text=True, encoding="utf-8")
            if result.returncode == 0:
                print("✅ Data_processing.py executed successfully!")
                return {"status": "success", "message": "Data processing completed successfully"}
            else:
                print("❌ Error executing Data_processing.py")
                print(result.stderr)
                return {"status": "error", "message": result.stderr}
        else:
            return {"status": "error", "message": "Data_processing.py not found"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------- 模型训练脚本执行函数 -----------------
def execute_model_training():
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'LSTM-GCN Model.py')
        python_executable = sys.executable

        if os.path.exists(script_path):
            print(f"⚙️ Executing training script: {script_path}")
            result = subprocess.run([python_executable, script_path], capture_output=True, text=True, encoding='utf-8')
            if result.returncode == 0:
                print("✅ LSTM-GCN Model executed successfully!")
                return {"status": "success", "message": "Model training completed successfully"}
            else:
                print("❌ Error executing LSTM-GCN Model.py")
                print(result.stderr)
                return {"status": "error", "message": result.stderr}
        else:
            return {"status": "error", "message": "LSTM-GCN Model.py not found"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------- 新增接口：仅训练模型 -----------------
@app.route('/api/train_model_only', methods=['POST'])
def train_model_only():
    try:
        training_result = execute_model_training()
        return jsonify({
            "status": training_result["status"],
            "message": training_result["message"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ----------------- 启动 Flask 应用 -----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
