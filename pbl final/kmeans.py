import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def visualize_kmeans_bitcoin(df, features):
    """
    Visualize K-means clustering steps for Bitcoin transaction data
    
    Args:
    df: Original DataFrame
    features: Preprocessed features for clustering
    """
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Use PCA for dimensionality reduction (2D visualization)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(20, 5))
    
    # 1. Original Data Visualization
    plt.subplot(141)
    plt.title('Step 1: Original Data Points\n(PCA Projection)')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Find elbow point
    plt.subplot(142)
    plt.title('Elbow Method for\nOptimal Clusters')
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    
    # Determine optimal clusters
    diffs = np.diff(inertias)
    optimal_clusters = np.where(diffs > np.mean(diffs))[0][-1] + 2
    
    # Perform K-means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    
    # 2. Initial Centroid Placement
    plt.subplot(143)
    plt.title(f'Step 2: Initial Centroids\n({optimal_clusters} Clusters)')
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    initial_centroids = kmeans.fit(X_scaled).cluster_centers_
    initial_centroids_2d = pca.transform(initial_centroids)
    plt.scatter(initial_centroids_2d[:, 0], initial_centroids_2d[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    
    # 3. Cluster Assignment
    plt.subplot(144)
    plt.title(f'Step 3: Final Clusters\n({optimal_clusters} Clusters)')
    labels = kmeans.fit_predict(X_scaled)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    final_centroids_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(final_centroids_2d[:, 0], final_centroids_2d[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.savefig('bitcoin_kmeans_visualization.png', dpi=300)
    plt.close()
    
    # Return clustering results for further analysis
    return {
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'optimal_clusters': optimal_clusters
    }

# Example usage similar to the original script
class BitcoinKMeansVisualizer(BitcoinFraudDetector):
    def analyze_kmeans(self, filepath):
        # Load data
        df = self.load_data(filepath)
        
        # Extract features
        features = self.extract_features(df)
        
        # Visualize K-means clustering
        clustering_results = visualize_kmeans_bitcoin(df, features)
        
        return clustering_results

# Run the visualization
if __name__ == "__main__":
    visualizer = BitcoinKMeansVisualizer()
    results = visualizer.analyze_kmeans("bitcoin_transactions.csv")
    print(f"Number of Clusters: {results['optimal_clusters']}")
