import streamlit as st
from sklearn.datasets import make_circles
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import graph_convexity as gc
import networkx as nx


st.title("Graph Convexity of circles")

n_samples = int(st.sidebar.number_input("Number of samples", key="moons", value=100))
random_state = int(st.sidebar.number_input("Random state", value=42))
noise = float(st.sidebar.number_input("Noise", value=0.05))
n_neighbors = int(st.sidebar.number_input("Number of neighbors", value=10))
factor = float(st.sidebar.number_input("Factor", value=0.5))


X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state, factor=factor)

col1, col2 = st.columns(2)

with col1:    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(X[:, 0], X[:, 1], c=y)
    st.pyplot(fig, clear_figure=True)

G = gc.construct_graph(X, n_neighbors=n_neighbors)
paths = gc.shortest_paths(G)
convexity_score = gc.convexity_score(paths, y)

with col2:
    fig, ax = plt.subplots()
    ax.set_title("Graph view")
    nx.draw(G, node_color=y, ax=ax, pos=X)
    st.pyplot(fig, clear_figure=True)

st.write(f"Convexity score: {convexity_score:.4f}")
