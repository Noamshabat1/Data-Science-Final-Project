import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import community as community_louvain
from sentence_transformers import SentenceTransformer
from cache_utils import GraphCacheManager
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
from wordcloud import WordCloud

SMALL_COMMUNITY_THRESHOLD = 20
MERGE_THRESHOLD = 0.3
TOP_N = 20

TITLE_FONT_SIZE = 12

STOP_WORDS = ['english', 'https', 'http', 'rt', 'co', 'is', 'the', 'and',
              'of', 'to', 'in', 'a', 'for', 'that', 'it', 'this', 'with',
              'on', 'as', 'at', 'by', 'an', 'be', 'are', 'was', 'were', 'from',
              'will', 'they', 'he', 'she', 'you', 'we', 'they', 'my', 'your',]

CHANGE_THRESHOLD = 0.2
SIMILARITY_THRESHOLD_DEFAULT = 0.7
TOP_K_DEFAULT = 10
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
SPRING_LAYOUT_SEED = 42
SPRING_LAYOUT_K = 2.0
NODE_SIZE_SCALING = 30
NODE_SIZE_MINIMUM = 10
WORDCLOUD_WIDTH = 1600
WORDCLOUD_HEIGHT = 800
WORDCLOUD_BACKGROUND_COLOR = 'white'
WORDCLOUD_COLORMAP = 'tab20'

class TweetGraphBuilder:
    def __init__(self, tweets, similarity_threshold=SIMILARITY_THRESHOLD_DEFAULT, top_k=TOP_K_DEFAULT, force_rebuild=False):
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
        pos = nx.spring_layout(self.graph, seed=SPRING_LAYOUT_SEED)
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
            reduced = TSNE(n_components=TSNE_COMPONENTS, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE).fit_transform(embeddings)
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

        reduced = self._get_tsne_reduced_embeddings()

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

        reduced = self._get_tsne_reduced_embeddings()

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
            label = self._extract_community_label(comm_id, texts)
            labels[comm_id] = label
            G.add_node(comm_id, pos=community_pos[comm_id], size=community_sizes[comm_id], label=label)

        for (id1, vec1), (id2, vec2) in combinations(community_pos.items(), 2):
            sim = cosine_similarity([vec1], [vec2])[0][0]
            if sim >= 0.6:
                G.add_edge(id1, id2, weight=sim)

        pos = nx.spring_layout(G, seed=SPRING_LAYOUT_SEED, k=SPRING_LAYOUT_K)
        # Ensure small communities are visible: minimum size 10, scale by 10
        sizes = [max(G.nodes[n]['size'], NODE_SIZE_MINIMUM) * NODE_SIZE_SCALING for n in G.nodes()]
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

    def visualize_community_wordcloud(self):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        community_texts = {}
        for node, comm_id in self.partition.items():
            community_texts.setdefault(comm_id, []).append(self.tweets.iloc[node]['text'])

        community_labels = {}
        for comm_id, texts in community_texts.items():
            label = self._extract_community_label(comm_id, texts)
            community_labels[label] = len(texts)

        wordcloud = WordCloud(width=WORDCLOUD_WIDTH, height=WORDCLOUD_HEIGHT, background_color=WORDCLOUD_BACKGROUND_COLOR, colormap=WORDCLOUD_COLORMAP)
        wordcloud.generate_from_frequencies(community_labels)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Community Word Cloud (Label Size Reflects Community Size)", fontsize=TITLE_FONT_SIZE)
        plt.show()

    def _extract_community_label(self, comm_id, texts):
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            max_features=1,
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b"
        )
        try:
            vectorizer.fit(texts)
            words = vectorizer.get_feature_names_out()
            return words[0] if len(words) > 0 else f"Comm {comm_id}"
        except:
            return f"Comm {comm_id}"

    def _get_tsne_reduced_embeddings(self):
        reduced = self.cache.load_tsne()
        if reduced is None:
            reduced = TSNE(n_components=TSNE_COMPONENTS, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE).fit_transform(self.embeddings)
            self.cache.save_tsne(reduced)
        return reduced

    def analyze_community_stock_relationship(self, stock):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        self.tweets['date'] = pd.to_datetime(self.tweets['timestamp']).dt.date
        self.tweets['community'] = self.tweets.index.map(self.partition)

        # Compute stock returns and significant change indicator
        stock = stock.copy()
        stock['return'] = stock['Close'].pct_change(periods=3)

        stock['significant_change'] = stock['return'].abs() >= CHANGE_THRESHOLD

        # Compute tweet counts by date/community, lagged by 1 day
        tweet_counts = self.tweets.groupby(['date', 'community']).size().unstack(fill_value=0)
        tweet_counts = tweet_counts.shift(1).dropna()  # lag tweets by 1 day
        stock['date'] = pd.to_datetime(stock['Date']).dt.date
        merged = pd.merge(tweet_counts, stock[['date', 'significant_change']], on='date')

        sig_days = merged[merged['significant_change']]
        non_sig_days = merged[~merged['significant_change']]

        mean_sig = sig_days.drop(columns=['significant_change', 'date']).mean()
        mean_non_sig = non_sig_days.drop(columns=['significant_change', 'date']).mean()

        diff = mean_sig - mean_non_sig
        top_n = 15
        top_diff = diff.abs().sort_values(ascending=False).head(top_n).index

        # Unique label assignment logic
        labels = {}
        used = set()
        for comm_id in top_diff:
            texts = [self.tweets.iloc[i]['text'] for i, cid in self.partition.items() if cid == comm_id]
            label = self._extract_community_label(comm_id, texts)
            base = label
            suffix = 1
            while label in used:
                label = f"{base} ({suffix})"
                suffix += 1
            used.add(label)
            labels[comm_id] = label

        import seaborn as sns
        import matplotlib.pyplot as plt
        heatmap_data = pd.DataFrame({
            'Increase in Tweets': mean_sig[top_diff] - mean_non_sig[top_diff],
            'label': [labels.get(i, f"Comm {i}") for i in top_diff]
        }).set_index('label').sort_values('Increase in Tweets', ascending=False)

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='YlGnBu',
            linewidths=0.5,
            cbar_kws={'label': 'Avg Tweet Count Diff (Sig - Non-Sig)'}
        )
        ax.set_title("Top Communities with Increased Tweets Before Drastic Stock Movements", fontsize=TITLE_FONT_SIZE)
        plt.yticks(rotation=0)
        plt.tight_layout()
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

    builder.visualize_community_wordcloud()

    builder.analyze_community_stock_relationship(stock)

if __name__ == '__main__':
    main()