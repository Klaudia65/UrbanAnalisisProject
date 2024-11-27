import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
data = pd.read_csv('data/data.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Clean the data
data = data.dropna()  # Remove rows with missing values

# Prepare the data
X = data.drop(columns=['Geography', 'city'])  # Replace 'Geography' with the name of your target column if needed

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to the data
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Visualize the PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', data=data, hue='city')  # Replace 'Geography' with a relevant column if needed
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)