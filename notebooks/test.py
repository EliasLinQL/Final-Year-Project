# import plotly.graph_objects as go
#
# fig = go.Figure(data=go.Scatter(y=[2, 3, 1]))
# fig.write_image("test.png")
# print("Test image saved successfully!")

import pandas as pd
import numpy as np
import networkx as nx
import os

# # Set the project root and data directory
# project_root = r"D:\Y3\FYP\Final-Year-Project"
# data_dir = os.path.join(project_root, "data")
#
# # Load graph edge list into a DataFrame
# graph_edges_file = os.path.join(data_dir, "graph_edges.csv")
# graph_edges = pd.read_csv(graph_edges_file)
#
# # Build the graph
# graph = nx.from_pandas_edgelist(graph_edges, source='source', target='target', edge_attr='weight')
#
# # Display basic graph information
# # degree
# print(nx.degree(graph))
# # 连通分量
# print(nx.connected_components(graph))
# # diameter
# print(nx.diameter(graph))
# # 度中心性
# print(nx.degree_centrality(graph))
# # 特征向量中心性
# print(nx.eigenvector_centrality(graph))
# # betweenness
# print(nx.betweenness_centrality(graph))
# # closeness
# print(nx.closeness_centrality(graph))
# # pagerank
# print(nx.pagerank(graph))
# # HITS
# print(nx.hits(graph))


# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# print(device)
# # 创建一个随机张量并在 GPU 上进行计算
# x = torch.rand(10000, 10000).to(device)
# y = torch.matmul(x, x)
# print(y)


import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.version.cuda)  # 检查 PyTorch 编译时的 CUDA 版本
