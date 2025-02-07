import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置项目路径
project_root = r"D:\Y3\FYP\Final-Year-Project\Final-Year-Project"
result_dir = os.path.join(project_root, "results")

# 1️⃣ 读取市场数据
btc_df = pd.read_csv(os.path.join(project_root, "data", "BTCUSDT_2021_2024.csv"))
btc_df['date'] = pd.to_datetime(btc_df['open_time'])  # 将时间列转换为日期时间格式
btc_df.set_index('date', inplace=True)  # 设置日期为索引
btc_df = btc_df[['open', 'high', 'low', 'close', 'volume']]  # 仅保留所需列

# 2️⃣ 读取模型预测结果
predictions = pd.read_csv(os.path.join(result_dir, "BTCUSDT_Predictions.csv"))
predictions['Date'] = pd.to_datetime(predictions['Date'])  # 修正列名并转换为日期时间格式
predictions.set_index('Date', inplace=True)  # 设置日期为索引

# 3️⃣ 每 4 小时交易一次
# 使用 resample 方法对预测结果进行重采样，每 4 小时取第一条预测数据
four_hour_predictions = predictions.resample('4h').first()

# **对齐预测数据和市场数据的时间范围**
aligned_index = four_hour_predictions.index.intersection(btc_df.index)  # 确保索引一致
btc_df = btc_df.loc[aligned_index]
four_hour_predictions = four_hour_predictions.loc[aligned_index]

# 转换预测价格为收益率
y_pred = four_hour_predictions['Predicted Price'].values / btc_df['close'].values - 1

# **确保数据长度对齐**
btc_prices = btc_df['close'].values
actual_returns = btc_df['close'].pct_change().values[1:]  # 去掉第一个 NaN 值
y_pred = y_pred[:len(btc_prices) - 1]  # 确保与 btc_prices[:-1] 长度一致

# **新增条件：仅在收益大于手续费时做多，看空时保持空仓**
FEE = 0.001  # 交易手续费 0.1%
btc_df = btc_df.iloc[:len(y_pred)]  # 修正 btc_df 长度
btc_df['y_pred'] = y_pred  # 添加预测收益率
btc_df['position'] = np.where(
    btc_df['y_pred'] > FEE, 1,  # 预测收益率大于手续费，看涨做多
    0  # 预测收益率小于等于手续费或看跌，保持空仓
)

# 统计交易次数
btc_df['trades'] = abs(btc_df['position'].diff().fillna(0))  # 计算交易次数

# 4️⃣ 计算交易策略收益（包含手续费）
btc_df['return'] = btc_df['close'].pct_change()  # 市场收益率
btc_df['strategy_return'] = (
    btc_df['position'].shift(1) * btc_df['return'] - FEE * btc_df['trades']
)  # 策略收益

# 5️⃣ 计算累计收益
btc_df['cumulative_market'] = (1 + btc_df['return']).cumprod()
btc_df['cumulative_strategy'] = (1 + btc_df['strategy_return']).cumprod()

# 6️⃣ 计算预测方向胜率
actual_directions = np.sign(btc_df['return'].iloc[:len(y_pred)].values)  # 市场实际涨跌方向
predicted_directions = np.sign(y_pred)  # 模型预测的方向
correct_predictions = (actual_directions == predicted_directions).sum()
total_predictions = len(predicted_directions)
win_rate = correct_predictions / total_predictions

# 7️⃣ 计算回测指标
max_drawdown = (btc_df['cumulative_strategy'].cummax() - btc_df['cumulative_strategy']).max()
sharpe_ratio = (btc_df['strategy_return'].mean() / btc_df['strategy_return'].std()) * (252 ** 0.5)

# 输出结果
print(f"Prediction Win Rate: {win_rate:.2f}")
print(f"Total Return: {btc_df['cumulative_strategy'].iloc[-1]:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Total Trades: {btc_df['trades'].sum()}\n")  # 输出交易次数

# 8️⃣ 绘制收益曲线
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
