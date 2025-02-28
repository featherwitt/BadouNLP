import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 进行K-Means聚类
k = 4
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

# 获取聚类标签和质心
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 计算每个簇的类内距离
intra_cluster_distances = []
for i in range(k):
    # 获取当前簇的样本
    cluster_samples = X[labels == i]
    # 计算当前簇的质心
    centroid = centroids[i]
    # 计算当前簇内所有样本到质心的距离
    distances = np.linalg.norm(cluster_samples - centroid, axis=1)
    # 计算类内距离（这里使用平均距离）
    intra_cluster_distance = np.mean(distances)
    intra_cluster_distances.append(intra_cluster_distance)

# 对簇进行排序，按照类内距离从小到大排序
sorted_clusters = sorted(range(k), key=lambda x: intra_cluster_distances[x])

# 输出排序结果
print("按照类内距离从小到大排序的簇索引：", sorted_clusters)
print("对应的类内距离：", [intra_cluster_distances[i] for i in sorted_clusters])
