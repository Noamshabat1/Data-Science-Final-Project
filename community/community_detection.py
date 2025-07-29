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

    def visualize_communities(self, title):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        reduced = self._get_tsne_reduced_embeddings()

        pos = {i: reduced[i] for i in range(len(reduced))}
        cmap = plt.get_cmap('viridis', max(self.partition.values()) + 1)
        colors = [cmap(self.partition[node]) for node in self.graph.nodes()]

        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=15, alpha=0.8)
        plt.title(title, fontsize=TITLE_FONT_SIZE)
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
        
        # Get top positive communities directly
        positive_diff = diff[diff > 0].sort_values(ascending=False).head(14)
        
        # Unique label assignment logic
        labels = {}
        used = set()
        for comm_id in positive_diff.index:
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
        
        plot_data = pd.DataFrame({
            'Increase in Tweets': positive_diff,
            'Community': [labels.get(i, f"Comm {i}") for i in positive_diff.index]
        }).sort_values('Increase in Tweets', ascending=True)  # ascending for horizontal bar

        plt.figure(figsize=(12, 8))
        bars = plt.barh(plot_data['Community'], plot_data['Increase in Tweets'], 
                       color='steelblue')
        
        plt.xlabel('Average Increase in Tweets Before Significant Stock Days', fontsize=11)
        plt.title("Communities with Increased Tweet Activity Before Major Stock Movements", fontsize=TITLE_FONT_SIZE)
        
        # Add y-axis label on the right
        ax = plt.gca()
        ax.yaxis.set_label_position('right')
        plt.ylabel('Number of Tweets', fontsize=11)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            plt.text(row['Increase in Tweets'] + 0.01, i, f'{row["Increase in Tweets"]:.2f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()

    def compute_modularity(self):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        modularity_score = community_louvain.modularity(self.partition, self.graph)
        print(f"Modularity Score: {modularity_score:.4f}")
        return modularity_score


    def plot_community_timeline_overlay(self, stock, top_n=5):
        if self.graph is None or not hasattr(self, 'partition'):
            raise ValueError("Graph not built or communities not detected.")

        self.tweets['date'] = pd.to_datetime(self.tweets['timestamp']).dt.date
        self.tweets['community'] = self.tweets.index.map(self.partition)

        stock = stock.copy()
        stock['return'] = stock['Close'].pct_change(periods=3)
        stock['significant_change'] = stock['return'].abs() >= CHANGE_THRESHOLD
        stock['date'] = pd.to_datetime(stock['Date']).dt.date

        tweet_counts = self.tweets.groupby(['date', 'community']).size().unstack(fill_value=0)
        tweet_counts = tweet_counts.shift(1).dropna()

        merged = pd.merge(tweet_counts, stock[['date', 'Close']], on='date', how='inner')

        # Compute impact score to select top communities
        stock['significant_change'] = stock['return'].abs() >= CHANGE_THRESHOLD
        sig_days = merged[merged['date'].isin(stock[stock['significant_change']]['date'])]
        non_sig_days = merged[~merged['date'].isin(stock[stock['significant_change']]['date'])]

        mean_sig = sig_days.drop(columns=['date', 'Close']).mean()
        mean_non_sig = non_sig_days.drop(columns=['date', 'Close']).mean()
        impact = (mean_sig - mean_non_sig).abs().sort_values(ascending=False)

        top_communities = impact.head(top_n).index

        # Build plot
        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(merged['date'], merged['Close'], color='black', label='Stock Price', linewidth=2)
        ax1.set_ylabel("Stock Price")
        ax1.set_xlabel("Date")
        ax1.set_title("Stock Price with Tweet Volume Overlay for Top Communities", fontsize=TITLE_FONT_SIZE + 2)

        ax2 = ax1.twinx()
        # Use a high-contrast and colorblind-friendly palette
        colors = sns.color_palette("tab10", len(top_communities)) if len(top_communities) <= 10 else sns.color_palette("tab20", len(top_communities))
        for i, comm_id in enumerate(top_communities):
            label = self._extract_community_label(comm_id, [self.tweets.iloc[j]['text'] for j, cid in self.partition.items() if cid == comm_id])
            ax2.plot(merged['date'], merged[comm_id], label=label, color=colors[i], linestyle='--')

        ax2.set_ylabel("Tweet Count")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.tight_layout()
        plt.show()

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'clean')
    
    tweets_path = os.path.join(data_dir, 'clean_musk_tweets.csv')
    stock_path = os.path.join(data_dir, 'clean_tesla_stock.csv')
    
    tweets = pd.read_csv(tweets_path)
    stock = pd.read_csv(stock_path)
    stock['Date'] = pd.to_datetime(stock['Date']).dt.strftime('%Y-%m-%d')
    tweets = tweets.dropna(subset=['text', 'timestamp']).reset_index(drop=True)
    return tweets, stock

def main():
    tweets, stock = load_data()
    builder = TweetGraphBuilder(tweets)
    builder.build_graph()
    builder.visualize_graph()

    builder.detect_communities_louvain()
    builder.visualize_communities(f"Community Detection with TSNE Layout ({len(set(builder.partition.values()))} communities)")

    builder.merge_similar_communities()
    builder.visualize_communities(f"Merged Communities ({len(set(builder.partition.values()))} communities)")
    builder.visualize_merged_community_graph()

    builder.analyze_community_stock_relationship(stock)

    builder.plot_community_timeline_overlay(stock)

    builder.compute_modularity()

if __name__ == '__main__':
    main()