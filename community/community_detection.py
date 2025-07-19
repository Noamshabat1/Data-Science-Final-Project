import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import community as community_louvain
from sentence_transformers import SentenceTransformer
from cache_utils import GraphCacheManager
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations

SMALL_COMMUNITY_THRESHOLD = 20
MERGE_THRESHOLD = 0.3
MIN_COMMUNITY_SIZE = 5
TOP_N = 20

TITLE_FONT_SIZE = 12

STOP_WORDS = ['english', 'https', 'http', 'rt', 'co', 'is', 'the', 'and',
              'of', 'to', 'in', 'a', 'for', 'that', 'it', 'this', 'with',
              'on', 'as', 'at', 'by', 'an', 'be', 'are', 'was', 'were', 'from',
              'will', 'they', 'he', 'she', 'you', 'we', 'they', 'my', 'your',]

class TweetGraphBuilder:
    def __init__(self, tweets, similarity_threshold=0.7, top_k=10, force_rebuild=False):
        self.tweets = tweets
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.graph = None
        self.cache = GraphCacheManager()
        self.force_rebuild = force_rebuild

    def build_graph(self):
        if not self.force_rebuild:
            cached_embeddings = self.cache.load_embeddings()
            cached_graph = self.cache.load_graph()
            if cached_embeddings is not None and cached_graph is not None:
                self.embeddings = cached_embeddings
                self.graph = cached_graph
                return

        texts = self.tweets['text'].astype(str).tolist()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
        self.embeddings = embeddings

        graph = nx.Graph()
        for i in range(len(texts)):
            graph.add_node(i)

        for i in range(len(texts)):
            sims = np.dot(embeddings, embeddings[i])
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(embeddings[i])
            similarity_scores = sims / norms
            sim_scores = list(enumerate(similarity_scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:self.top_k + 1]
            for j, sim in sim_scores:
                if sim >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=sim)

        self.graph = graph
        self.cache.save_embeddings(embeddings)
        self.cache.save_graph(graph)

    def visualize_raw_graph(self):
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue', node_size=15, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', alpha=0.3)
        plt.title("Raw Tweet Similarity Graph (Before Community Detection)", fontsize=TITLE_FONT_SIZE)
        plt.show()

    def visualize_graph(self):
        if self.graph is None:
            raise ValueError("Graph not built yet.")

        reduced = self.cache.load_tsne()
        if reduced is None:
            embeddings = self.embeddings
            reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
            self.cache.save_tsne(reduced)

        pos = {i: reduced[i] for i in range(len(reduced))}
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue', node_size=15, alpha=0.8)
        plt.title("Tweet Similarity Graph using TSNE Node Embeddings", fontsize=TITLE_FONT_SIZE)
        plt.show()

    def detect_communities_louvain(self):
        cached_partition = self.cache.load_partition("louvain")
        if cached_partition is not None:
            self.partition = cached_partition
            self.partition_label = "louvain"
            return

        if self.graph is None:
            raise ValueError("Graph not built yet.")
        # Lower resolution merges more nodes into fewer communities
        self.partition = community_louvain.best_partition(self.graph)
        self.partition_label = "louvain"
        self.cache.save_partition(self.partition, "louvain")

    def merge_similar_communities(self, threshold=0.3):
        cached_partition = self.cache.load_partition("merged")
        if cached_partition is not None:
            self.partition = cached_partition
            self.partition_label = "merged"
            return

        community_vectors = {}
        for node, comm_id in self.partition.items():
            community_vectors.setdefault(comm_id, []).append(self.embeddings[node])
        centroids = {
            comm_id: np.mean(vectors, axis=0)
            for comm_id, vectors in community_vectors.items()
        }

        merged = True
        while merged:
            merged = False
            comm_ids = list(centroids.keys())
            vectors = [centroids[comm_id] for comm_id in comm_ids]
            sim_matrix = cosine_similarity(vectors)

            merge_map = {}

            for i in range(len(comm_ids)):
                for j in range(i + 1, len(comm_ids)):
                    id_i = comm_ids[i]
                    id_j = comm_ids[j]
                    size_i = len(community_vectors[id_i])
                    size_j = len(community_vectors[id_j])
                    if (size_i < SMALL_COMMUNITY_THRESHOLD or size_j < SMALL_COMMUNITY_THRESHOLD) and sim_matrix[i, j] >= MERGE_THRESHOLD:
                        src, dst = (id_j, id_i) if size_j < size_i else (id_i, id_j)
                        merge_map[src] = dst
                        merged = True

            if merged:
                new_partition = {}
                for node, comm_id in self.partition.items():
                    while comm_id in merge_map:
                        comm_id = merge_map[comm_id]
                    new_partition[node] = comm_id
                self.partition = new_partition

                community_vectors = {}
                for node, comm_id in self.partition.items():
                    community_vectors.setdefault(comm_id, []).append(self.embeddings[node])
                centroids = {
                    comm_id: np.mean(vectors, axis=0)
                    for comm_id, vectors in community_vectors.items()
                }
                if len(centroids) <= 300:
                    break

        self.partition_label = "merged"
        self.cache.save_partition(self.partition, "merged")

    def visualize_communities(self):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        reduced = self.cache.load_tsne()
        if reduced is None:
            reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(self.embeddings)
            self.cache.save_tsne(reduced)

        pos = {i: reduced[i] for i in range(len(reduced))}
        cmap = plt.get_cmap('viridis', max(self.partition.values()) + 1)
        colors = [cmap(self.partition[node]) for node in self.graph.nodes()]

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=15, alpha=0.8)
        method_name = 'Louvain' if 'louvain' in self.partition_label.lower() else 'Label Propagation'
        plt.title(f"{method_name} Community Detection with TSNE Layout ({len(set(self.partition.values()))} communities)", fontsize=TITLE_FONT_SIZE)
        plt.show()

    def visualize_merged_community_graph(self):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        reduced = self.cache.load_tsne()
        if reduced is None:
            reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(self.embeddings)
            self.cache.save_tsne(reduced)

        community_pos = {}
        community_sizes = {}
        for node, comm_id in self.partition.items():
            community_pos.setdefault(comm_id, []).append(reduced[node])

        for comm_id in community_pos:
            points = np.array(community_pos[comm_id])
            community_pos[comm_id] = points.mean(axis=0)
            community_sizes[comm_id] = len(points)

        G = nx.Graph()
        labels = {}
        for comm_id, nodes in community_pos.items():
            texts = [self.tweets.iloc[node]['text'] for node, cid in self.partition.items() if cid == comm_id]
            vectorizer = TfidfVectorizer(
                stop_words=STOP_WORDS,
                max_features=1,
                token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b"
            )
            try:
                vectorizer.fit(texts)
                words = vectorizer.get_feature_names_out()
                label = words[0] if len(words) > 0 else f"Comm {comm_id}"
            except:
                label = f"Comm {comm_id}"
            labels[comm_id] = label
            G.add_node(comm_id, pos=community_pos[comm_id], size=community_sizes[comm_id], label=label)

        for (id1, vec1), (id2, vec2) in combinations(community_pos.items(), 2):
            sim = cosine_similarity([vec1], [vec2])[0][0]
            if sim >= 0.6:
                G.add_edge(id1, id2, weight=sim)

        pos = nx.spring_layout(G, seed=42, k=2.0)
        # Ensure small communities are visible: minimum size 10, scale by 10
        sizes = [max(G.nodes[n]['size'], 10) * 30 for n in G.nodes()]
        plt.figure(figsize=(20, 18))
        # Use tab20 colormap for clearer color separation
        cmap = plt.cm.get_cmap('tab20', len(G.nodes()) + 1)
        node_colors = [cmap(i) for i in range(len(G.nodes()))]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors)
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(
            G,
            pos,
            labels=node_labels,
            font_size=20,
            font_color='white',
            font_weight='bold',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title("Community Graph After Merge (Node Size Reflects Community Size)", fontsize=24)
        plt.axis('off')
        plt.show()


def load_data():
    tweets = pd.read_csv('../data/clean/clean_musk_tweets.csv')
    stock = pd.read_csv('../data/clean/clean_stock.csv')
    stock['Date'] = pd.to_datetime(stock['Date']).dt.strftime('%Y-%m-%d')
    tweets = tweets.dropna(subset=['text', 'timestamp']).reset_index(drop=True)
    return tweets, stock

def main():
    tweets, stock = load_data()
    builder = TweetGraphBuilder(tweets)
    builder.build_graph()
    builder.visualize_graph()

    builder.detect_communities_louvain()
    builder.visualize_communities()

    builder.merge_similar_communities()
    builder.visualize_communities()
    builder.visualize_merged_community_graph()

if __name__ == '__main__':
    main()