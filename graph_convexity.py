import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph


def construct_graph(X, n_neighbors=10):
    G = nx.from_numpy_array(kneighbors_graph(X, n_neighbors, mode="distance", metric="euclidean").toarray())
    return G

def shortest_paths(G):
    nodes = list(G.nodes())

    paths = []
    for n in nodes:
        for m in nodes:
            if n != m:
                if nx.has_path(G, n, m):
                    paths.append(nx.dijkstra_path(G, n, m))
                else:
                    # append -1 to indicate no existing path
                    paths.append([n, m, -1])
    return paths

def relevant_path_labels(paths, labels):
    path_labels = []
    for path in paths:
        no_path_exists = path[-1] == -1
        ends_have_same_label = labels[path[0]] == labels[path[-1]]

        if no_path_exists and ends_have_same_label:
            path_labels.append([-1])
        else:
            path_labels.append(labels[path])
    return path_labels

def convexity_score(paths, labels):
    path_labels_list = relevant_path_labels(paths, labels)

    path_scores = []
    for path_labels in path_labels_list:
        no_path_exists = path_labels[0] == -1
        ends_have_same_label = path_labels[0] == path_labels[-1]

        if no_path_exists:
            path_scores.append(0.0)
        if len(path_labels) == 2 and ends_have_same_label:
            path_scores.append(1.0)
        if len(path_labels) > 2 and ends_have_same_label:
            path_score = np.mean(path_labels[1:-1] == path_labels[0])
            path_scores.append(path_score)

    return np.mean(path_scores)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    G = construct_graph(X, n_neighbors=10)
    paths = shortest_paths(G)
    convexity_score = convexity_score(paths, y)
    print(f"Convexity score: {convexity_score}")
    print(f"Number of paths: {len(paths)}")