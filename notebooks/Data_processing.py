import os
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# File paths
project_root = r"D:\Y3\FYP\Final-Year-Project"
data_dir = os.path.join(project_root, "data")  # Path to the data folder

# Load daily closing price data
file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

# Extract closing prices for all cryptocurrencies
crypto_prices = {}
for file in file_list:
    symbol = file.split('_')[0]  # Extract the trading pair name
    df = pd.read_csv(os.path.join(data_dir, file))
    df['date'] = pd.to_datetime(df['open_time'])  # Convert date format
    df.set_index('date', inplace=True)  # Set date as index
    crypto_prices[symbol] = df['close']

# Create a DataFrame with closing prices for all cryptocurrencies
prices_df = pd.DataFrame(crypto_prices)

# Calculate daily returns (percentage changes)
returns_df = prices_df.pct_change().dropna()

# ----------------- Feature Engineering -----------------
def generate_temporal_features(prices_df, returns_df, lookback_days=5, ma_windows=[5, 10]):
    """
    Generate temporal features for each cryptocurrency, including:
    - Returns over the last x days
    - Moving Averages (MAs)

    Parameters:
    prices_df: pd.DataFrame, columns represent the closing prices of each cryptocurrency
    returns_df: pd.DataFrame, columns represent the daily returns of each cryptocurrency
    lookback_days: int, the number of days to look back for returns
    ma_windows: list, window sizes for moving averages

    Returns:
    features_df: pd.DataFrame, containing temporal features for all cryptocurrencies
    """
    features = {}

    for symbol in prices_df.columns:
        symbol_features = pd.DataFrame(index=prices_df.index)

        # Returns over the last x days
        for i in range(1, lookback_days + 1):
            symbol_features[f'{symbol}_return_lag_{i}'] = returns_df[symbol].shift(i)

        # Moving Averages
        for window in ma_windows:
            symbol_features[f'{symbol}_ma_{window}'] = prices_df[symbol].rolling(window).mean()

        # Combine all features
        features[symbol] = symbol_features

    # Concatenate features of all cryptocurrencies
    features_df = pd.concat(features.values(), axis=1)

    # Remove rows with NaN values (caused by rolling window)
    features_df = features_df.dropna()

    return features_df

# Define parameters for feature engineering
lookback_days = 5  # Returns over the last 5 days
ma_windows = [9, 25]  # Moving averages for 9 and 25 days

# Generate temporal features
features_df = generate_temporal_features(prices_df, returns_df, lookback_days, ma_windows)
print("Generated Features:")
print(features_df.head())

# ----------------- Correlation Matrix Calculation -----------------
# Calculate correlation matrix
correlation_matrix = returns_df.corr()

# Extract non-diagonal elements
non_diag_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]  # Extract upper triangle non-diagonal elements
sorted_values = np.sort(non_diag_values)[::-1]  # Sort in descending order

# Find an appropriate threshold
def find_threshold(correlation_matrix, sorted_values):
    for threshold in sorted_values:
        # Build the graph
        edges = []
        for i in correlation_matrix.index:
            for j in correlation_matrix.columns:
                if i != j and correlation_matrix.loc[i, j] >= threshold:
                    edges.append((i, j, correlation_matrix.loc[i, j]))

        # Create a graph using NetworkX
        G = nx.Graph()
        G.add_nodes_from(correlation_matrix.columns)
        G.add_weighted_edges_from([(u, v, w) for u, v, w in edges])

        # Check the degree of each node
        degrees = dict(G.degree())
        min_degree = min(degrees.values()) if degrees else 0

        # Return the current threshold if all nodes have at least one edge
        if min_degree >= 1:
            return threshold, G
    return 0, None  # Return 0 and an empty graph if no threshold is found

# Find the threshold based on the correlation matrix
threshold, G = find_threshold(correlation_matrix, sorted_values)
print(f"Selected Threshold: {threshold}")

# Output graph information
print("Graph Info:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print("Edges:", list(G.edges(data=True)))

# # graph Visualization
# import matplotlib.pyplot as plt
# pos = nx.spring_layout(G, seed=42)
# plt.figure(figsize=(12, 8))
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
#
# # Add an edge weight label
# edge_labels = nx.get_edge_attributes(G, 'weight')
# edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# ----------------- 3D Visualization -----------------
# Generate a 3D layout and visualize the graph
pos = nx.spring_layout(G, dim=3, seed=42)  # Generate a 3D layout

# Extract positions of nodes and edges
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
    edge_x += [x0, x1, None]  # Add the start and end of each edge
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

    # Midpoint coordinates for edge labels
    edge_labels_x.append((x0 + x1) / 2)
    edge_labels_y.append((y0 + y1) / 2)
    edge_labels_z.append((z0 + z1) / 2)
    edge_weights.append(f'{edge[2]["weight"]:.2f}')  # Keep two decimal places

# Create a Plotly graph object
fig = go.Figure()

# Add 3D edges
fig.add_trace(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines',
    line=dict(color='gray', width=1),
    hoverinfo='none'
))

# Add 3D nodes
fig.add_trace(go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers+text',
    marker=dict(size=10, color='lightblue'),
    text=list(G.nodes),
    textposition="top center",
    hoverinfo='text'
))

# Add edge weight labels
fig.add_trace(go.Scatter3d(
    x=edge_labels_x, y=edge_labels_y, z=edge_labels_z,
    mode='text',
    text=edge_weights,
    textfont=dict(color='red', size=10),
    hoverinfo='none'
))

# Configure the layout
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

# Save the graph as an HTML file
output_html_path = os.path.join(project_root, "results", "cryptocurrency_correlation_graph_3d.html")
fig.write_html(output_html_path)
print(f"3D Graph saved as: {output_html_path}")

# Save the features
features_df.to_csv(os.path.join(project_root, "data", "features.csv"), index=True)
print("Features saved to 'features.csv'")

# Save the correlation matrix
correlation_matrix.to_csv(os.path.join(project_root, "data", "correlation_matrix.csv"))
print("Correlation matrix saved to 'correlation_matrix.csv'")

# Save the graph (edge list)
edges = [(u, v, w['weight']) for u, v, w in G.edges(data=True)]
edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
edges_df.to_csv(os.path.join(project_root, "data", "graph_edges.csv"), index=False)
print("Graph edges saved to 'graph_edges.csv'")

# Display the graph
fig.show()
