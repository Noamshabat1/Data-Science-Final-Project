import os
import pickle
import numpy as np


class GraphCacheManager:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir

    def save_embeddings(self, embeddings):
        np.save(os.path.join(self.base_dir, "embeddings.npy"), embeddings)

    def load_embeddings(self):
        path = os.path.join(self.base_dir, "embeddings.npy")
        return np.load(path) if os.path.exists(path) else None

    def save_graph(self, graph):
        with open(os.path.join(self.base_dir, "graph.gpickle"), "wb") as f:
            pickle.dump(graph, f)

    def load_graph(self):
        path = os.path.join(self.base_dir, "graph.gpickle")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def save_tsne(self, reduced):
        np.save(os.path.join(self.base_dir, "tsne.npy"), reduced)

    def load_tsne(self):
        path = os.path.join(self.base_dir, "tsne.npy")
        return np.load(path) if os.path.exists(path) else None

    def save_partition(self, partition, label):
        with open(os.path.join(self.base_dir, f"partition_{label}.pkl"), "wb") as f:
            pickle.dump(partition, f)

    def load_partition(self, label):
        path = os.path.join(self.base_dir, f"partition_{label}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def clear_partition(self, label):
        path = os.path.join(self.base_dir, f"partition_{label}.pkl")
        if os.path.exists(path):
            os.remove(path)
