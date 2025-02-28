from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import pandas as pd
import requests
import subprocess  # 用于执行 Data_processing.py

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 允许跨域请求

# Binance API Base URL
BASE_URL = 'https://api.binance.com'

# 数据存储目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')

# 确保 data 目录存在
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


@app.route('/api/fetch_crypto_data', methods=['POST'])
def fetch_crypto_data():
    """
    接收前端请求，获取加密货币数据，并在成功后调用 Data_processing.py 处理数据。
    """
    try:
        request_data = request.json
        print("✅ Received Request Data:", request_data)

        start_date = request_data.get("start_date", "2021-01-01")
        end_date = request_data.get("end_date", "2025-01-01")
        symbols = request_data.get("symbols", ["BTCUSDT", "ETHUSDT"])

        start_time = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')) * 1000)
        end_time = int(time.mktime(time.strptime(end_date, '%Y-%m-%d')) * 1000)

        print(f"📅 Fetching data from {start_date} to {end_date} for symbols: {symbols}")

        # 采集数据
        file_paths = fetch_and_save_crypto_data(symbols, start_time, end_time)

        # 调用数据处理
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



def fetch_and_save_crypto_data(symbols, start_time, end_time):
    """
    根据提供的交易对和时间范围，从 Binance API 获取数据，并存储为 CSV 文件
    """
    limit = 1000  # 最大请求限制
    file_paths = []  # 记录已保存的文件

    # 转换时间戳为 YYYY-MM-DD 格式
    start_date_str = time.strftime('%Y-%m-%d', time.gmtime(start_time / 1000))
    end_date_str = time.strftime('%Y-%m-%d', time.gmtime(end_time / 1000))

    for symbol in symbols:
        # 更新 CSV 文件名格式
        output_file = os.path.join(data_dir, f'{symbol}_{start_date_str}_{end_date_str}.csv')
        final_df = pd.DataFrame()
        current_start_time = start_time

        while current_start_time < end_time:
            url = (
                f"{BASE_URL}/api/v3/klines?symbol={symbol}&interval=4h"
                f"&limit={limit}&startTime={current_start_time}&endTime={end_time}"
            )
            print(f"🔍 Fetching data for {symbol} from: {url}")
            response = requests.get(url)
            data = response.json()

            if not data or isinstance(data, dict):  # 如果 API 返回错误信息
                print(f"⚠️ No data returned for {symbol}")
                break  # 退出循环

            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

            final_df = pd.concat([final_df, df], ignore_index=True)

            # 更新起始时间
            current_start_time = int(df['open_time'].iloc[-1].timestamp() * 1000) + 1

        if not final_df.empty:
            final_df.to_csv(output_file, index=False)
            file_paths.append(output_file)
            print(f"✅ Data for {symbol} successfully saved to {output_file}")

    return file_paths  # 返回所有生成的 CSV 文件路径


def execute_data_processing():
    """
    运行 Data_processing.py 脚本，并返回执行结果。
    """
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'Data_processing.py')

        if os.path.exists(script_path):
            print(f"⚙️ Executing {script_path} ...")
            result = subprocess.run(["python", script_path], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Data_processing.py executed successfully!")
                print(result.stdout)  # 打印输出结果
                return {"status": "success", "message": "Data processing completed successfully"}
            else:
                print("❌ Error executing Data_processing.py")
                print(result.stderr)  # 打印错误信息
                return {"status": "error", "message": result.stderr}
        else:
            print(f"❌ Error: {script_path} not found!")
            return {"status": "error", "message": "Data_processing.py not found"}

    except Exception as e:
        print(f"🔥 Execution Error: {str(e)}")
        return {"status": "error", "message": str(e)}



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
