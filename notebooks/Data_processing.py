import os
import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go

# ------------------- Initialize Paths -------------------
data_dir = "../data"
results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

# ------------------- Load CSV Data -------------------
file_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
print("📂 CSV file list:", file_list)

crypto_prices = {}
for file in file_list:
    symbol = file.split('_')[0]
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.strip()

    if 'open_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['open_time'])
        df.set_index('datetime', inplace=True)
        crypto_prices[symbol] = df['close']
    else:
        print(f"⚠️ Warning: File `{file}` is missing 'open_time' column. Skipping.")

prices_df = pd.DataFrame(crypto_prices)

# ------------------- Calculate Returns -------------------
returns_df = prices_df.pct_change(fill_method=None).dropna()

# ------------------- Feature Engineering -------------------
def generate_temporal_features(prices_df, returns_df, lookback_days=6, ma_windows=[18, 54]):
    features = {}
    for symbol in prices_df.columns:
        symbol_features = pd.DataFrame(index=prices_df.index)
        for i in range(1, lookback_days + 1):
            symbol_features[f'{symbol}_return_lag_{i}'] = returns_df[symbol].shift(i)
        for window in ma_windows:
            symbol_features[f'{symbol}_ma_{window}'] = prices_df[symbol].rolling(window).mean()
        features[symbol] = symbol_features
    features_df = pd.concat(features.values(), axis=1).dropna()
    return features_df

features_df = generate_temporal_features(prices_df, returns_df)

print("✅ Features generated successfully:")
print(features_df.head())

# ------------------- Build Correlation Graph -------------------
correlation_matrix = returns_df.corr()
non_diag_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]
sorted_values = np.sort(non_diag_values)[::-1]

def find_threshold(corr_matrix, sorted_vals):
    for threshold in sorted_vals:
        edges = []
        for i in corr_matrix.index:
            for j in corr_matrix.columns:
                if i != j and corr_matrix.loc[i, j] >= threshold:
                    edges.append((i, j, corr_matrix.loc[i, j]))
        G = nx.Graph()
        G.add_nodes_from(corr_matrix.columns)
        G.add_weighted_edges_from(edges)
        degrees = dict(G.degree())
        min_degree = min(degrees.values()) if degrees else 0
        if min_degree >= 1:
            return threshold, G
    return 0, None

threshold, G = find_threshold(correlation_matrix, sorted_values)
print(f"✅ Selected correlation threshold: {threshold}")

# ------------------- Visualize & Save Graph -------------------
if G is not None:
    print("📊 Graph structure details:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Edge details:", list(G.edges(data=True)))

    pos = nx.spring_layout(G, dim=3, seed=42)
    x_nodes = [pos[node][0] for node in G.nodes]
    y_nodes = [pos[node][1] for node in G.nodes]
    z_nodes = [pos[node][2] for node in G.nodes]

    edge_x, edge_y, edge_z = [], [], []
    edge_labels_x, edge_labels_y, edge_labels_z, edge_weights = [], [], [], []

    for edge in G.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        edge_labels_x.append((x0 + x1) / 2)
        edge_labels_y.append((y0 + y1) / 2)
        edge_labels_z.append((z0 + z1) / 2)
        edge_weights.append(f'{edge[2]["weight"]:.2f}')

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=10, color='lightblue'),
        text=list(G.nodes),
        textposition="top center",
        hoverinfo='text'
    ))

    fig.add_trace(go.Scatter3d(
        x=edge_labels_x, y=edge_labels_y, z=edge_labels_z,
        mode='text',
        text=edge_weights,
        textfont=dict(color='red', size=10),
        hoverinfo='none'
    ))

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

    html_path = os.path.join(results_dir, "cryptocurrency_correlation_graph_3d.html")
    fig.write_html(html_path)
    print(f"✅ Graph saved as: {html_path}")

    # Save edges
    edges = [(u, v, w['weight']) for u, v, w in G.edges(data=True)]
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
    edges_df.to_csv(os.path.join(data_dir, "graph_edges.csv"), index=False)
    print("✅ Graph edges saved as graph_edges.csv")

    # Show figure
    fig.show()
else:
    print("⚠️ Warning: Graph G is None. No sufficient structure found. Skipping visualization and save.")

# ------------------- Save Features & Correlation Matrix -------------------
features_df.to_csv(os.path.join(data_dir, "features.csv"), index=True)
correlation_matrix.to_csv(os.path.join(data_dir, "correlation_matrix.csv"))
print("✅ Features and correlation matrix saved successfully.")
