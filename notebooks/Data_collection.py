import os
import time
import pandas as pd
import requests

# Binance API基础URL
BASE_URL = 'https://api.binance.com'

# 时间设置：从2021年1月1日到2024年11月1日（时间戳单位为毫秒）
start_time = int(time.mktime(time.strptime('2021-01-01', '%Y-%m-%d')) * 1000)
end_time = int(time.mktime(time.strptime('2024-11-01', '%Y-%m-%d')) * 1000)

# 每次请求的最大数据条数
limit = 1000

# 获取当前脚本所在目录（notebooks文件夹）
notebooks_dir = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（notebooks的上一级目录）
project_root = os.path.abspath(os.path.join(notebooks_dir, '..'))

# data文件夹路径
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 交易对列表
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'TRXUSDT']

# 遍历每个交易对获取数据
for symbol in symbols:
    output_file = os.path.join(data_dir, f'{symbol}_2021_2024.csv')

    # 初始化DataFrame
    final_df = pd.DataFrame()

    # 循环获取数据
    current_start_time = start_time
    while current_start_time < end_time:
        url = (
                BASE_URL +
                f'/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}&startTime={current_start_time}&endTime={end_time}'
        )
        print(f"Fetching data for {symbol} from: {url}")
        response = requests.get(url)
        data = response.json()

        if len(data) == 0:
            break  # 如果没有数据，退出循环

        # 将数据转换为DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # 处理时间戳为日期
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # 选择必要的列
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

        # 将数据追加到最终的DataFrame
        final_df = pd.concat([final_df, df], ignore_index=True)

        # 更新start_time为最后一个记录的open_time
        current_start_time = int(df['open_time'].iloc[-1].timestamp() * 1000) + 1

    # 保存到CSV文件
    final_df.to_csv(output_file, index=False)
    print(f"Data for {symbol} successfully saved to {output_file}")

    # 打印数据预览
    print(final_df.head())
