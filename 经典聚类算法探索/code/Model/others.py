from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering

# 层次聚类
def hierarchical_clustering(X):
    model = AgglomerativeClustering(n_clusters=10)
    labels = model.fit_predict(X)
    return labels


# 谱聚类
def spectral_clustering(X):
    model = SpectralClustering(n_clusters=10, affinity="nearest_neighbors")
    labels = model.fit_predict(X)
    return labels
