import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

data = pd.read_csv('data/data.csv')
print(data.head())
print(data.info())
print(data.describe())

data = data.dropna()  
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']

# Encode the 'city' column
encoder = OneHotEncoder()
city_encoded = encoder.fit_transform(data[['city']]).toarray()
encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['city']))
data = pd.concat([data, encoded_df], axis=1)
data.drop('city', axis=1, inplace=True)

X = data.drop(columns=['Geography']) 

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
silhouette_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

#Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', color='green')
plt.title('Silhouette Scores for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Perform KMeans clustering
optimal_clusters = 4 
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

data['cluster'] = kmeans.labels_
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]
centroids = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))
for i in range(data.shape[0]):
    plt.text(
        x=data['PCA1'].iloc[i],
        y=data['PCA2'].iloc[i],
        s=str(i),  # Use index or a specific column
        fontsize=8,
        alpha=0.7
    )

explained_variance = pca.explained_variance_ratio_

for i, centroid in enumerate(centroids):
    plt.scatter(*centroid, color='red', marker='X', s=200)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=data['cluster'], palette='viridis', s=100, alpha=0.8)
plt.title('KMeans Clustering with PCA')
plt.xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}% Variance)')
plt.ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}% Variance)')
plt.show()

# Heatmap of feature correlation (excluding one-hot columns)
numeric_features = data.select_dtypes(include=[float, int]).drop(columns=encoded_df.columns)
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation (Numeric Features Only)')
plt.show()

# Describe clusters
numeric_columns = data.select_dtypes(include=[float, int]).columns
cluster_summary = data.groupby('cluster')[numeric_columns].mean()
print("Cluster Summary:")
print(cluster_summary)

data.to_csv('clustered_data.csv', index=False)
