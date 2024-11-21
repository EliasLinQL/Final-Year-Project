import os
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# 数据文件路径
project_root = r"D:\Y3\FYP\Final-Year-Project"
data_dir = os.path.join(project_root, "data")  # 数据文件夹路径

# 加载每日收盘价数据
file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

# 提取所有加密货币的收盘价
crypto_prices = {}
for file in file_list:
    symbol = file.split('_')[0]  # 提取交易对名称
    df = pd.read_csv(os.path.join(data_dir, file))
    df['date'] = pd.to_datetime(df['open_time'])  # 转换日期格式
    df.set_index('date', inplace=True)  # 将日期设置为索引
    crypto_prices[symbol] = df['close']

# 创建包含所有加密货币收盘价的DataFrame
prices_df = pd.DataFrame(crypto_prices)

# 计算每日收益率（百分比变化）
returns_df = prices_df.pct_change().dropna()

# 计算相关性矩阵
correlation_matrix = returns_df.corr()

# 提取非对角元素
non_diag_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]  # 提取上三角非对角线元素
sorted_values = np.sort(non_diag_values)[::-1]  # 降序排序

# 寻找合适的阈值
def find_threshold(correlation_matrix, sorted_values):
    for threshold in sorted_values:
        # 构建图
        edges = []
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i != j and correlation_matrix.loc[i, j] >= threshold:
                    edges.append((i, j, correlation_matrix.loc[i, j]))

        # 使用NetworkX创建图
        G = nx.Graph()
        G.add_nodes_from(correlation_matrix.columns)
        G.add_weighted_edges_from([(u, v, w) for u, v, w in edges])

        # 检查每个节点的度数
        degrees = dict(G.degree())
        min_degree = min(degrees.values()) if degrees else 0

        # 如果每个节点至少有一条边，则返回当前阈值
        if min_degree >= 1:
            return threshold, G
    return 0, None  # 如果所有阈值都不满足，返回0和空图

# 根据相关性矩阵找到合适的阈值
threshold, G = find_threshold(correlation_matrix, sorted_values)
print(f"Selected Threshold: {threshold}")

# 输出结果
print(f"Selected Threshold: {threshold}")
print("Graph Info:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("Edges:", list(G.edges(data=True)))

# # 可视化图
# import matplotlib.pyplot as plt
# pos = nx.spring_layout(G, seed=42)
# plt.figure(figsize=(12, 8))
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
#
# # 添加边权重标签
# edge_labels = nx.get_edge_attributes(G, 'weight')
# edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# 使用 NetworkX 生成 3D 布局
pos = nx.spring_layout(G, dim=3, seed=42)  # 生成3D布局

# 提取节点和边的位置
x_nodes = [pos[node][0] for node in G.nodes]
y_nodes = [pos[node][1] for node in G.nodes]
z_nodes = [pos[node][2] for node in G.nodes]

edge_x = []
edge_y = []
edge_z = []
edge_labels_x = []
edge_labels_y = []
edge_labels_z = []
edge_weights = []

for edge in G.edges(data=True):
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]  # 添加边的起点和终点
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

    # 中点坐标用于显示边权重标签
    edge_labels_x.append((x0 + x1) / 2)
    edge_labels_y.append((y0 + y1) / 2)
    edge_labels_z.append((z0 + z1) / 2)
    edge_weights.append(f'{edge[2]["weight"]:.2f}')  # 保留两位小数

# 创建 Plotly 图表对象
fig = go.Figure()

# 添加边的3D线条
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=1),
    hoverinfo='none'
))

# 添加节点的3D散点
fig.add_trace(go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(size=10, color='lightblue'),
    text=list(G.nodes),
    textposition="top center",
    hoverinfo='text'
))

# 添加边权重标签
fig.add_trace(go.Scatter3d(
    x=edge_labels_x, y=edge_labels_y, z=edge_labels_z,
    mode='text',
    text=edge_weights,
    textfont=dict(color='red', size=10),
    hoverinfo='none'
))

# 设置图表布局
fig.update_layout(
    title="Cryptocurrency Correlation Graph (3D)",
    showlegend=False,
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

# 保存图表为 HTML 文件
output_html_path = os.path.join(project_root, "results", "cryptocurrency_correlation_graph_3d.html")
fig.write_html(output_html_path)
print(f"3D Graph saved as: {output_html_path}")

# 显示图表
fig.show()
