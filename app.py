import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import graph_convexity as gc
import networkx as nx

st.title("Graph Convexity of blobs")

n_samples = int(st.sidebar.number_input("Number of samples", key="blobs", value=100))
n_centers = int(st.sidebar.number_input("Number of centers", value=3))
n_neighbors = int(st.sidebar.number_input("Number of neighbors", value=10))
n_features = int(st.sidebar.number_input("Number of features", value=768))
cluster_std = float(st.sidebar.number_input("Cluster standard deviation", value=1.0))
random_state = int(st.sidebar.number_input("Random state", value=42))

col1, col2 = st.columns(2)

X, y = make_blobs(n_samples=n_samples, 
                n_features=n_features,
                centers=n_centers,
                cluster_std=cluster_std, 
                random_state=random_state)

with col1:    
    X_embedded = TSNE(n_components=2).fit_transform(X)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_title("2D tSNE embedding")
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)
    st.pyplot(fig, clear_figure=True)

G = gc.construct_graph(X, n_neighbors=n_neighbors)
paths = gc.shortest_paths(G)
convexity_score = gc.convexity_score(paths, y)


with col2:
    fig, ax = plt.subplots()
    ax.set_title("Graph view")
    nx.draw(G, node_color=y, ax=ax)
    st.pyplot(fig, clear_figure=True)

st.write(f"Convexity score: {convexity_score:.4f}")
