import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_outliers_iqr(data): # to see if a value is an outlier or not
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_iforest(dataset): # to see if a value is an outlier or not
    clf = IsolationForest(contamination=0.05)  # 5% of the data are outliers
    clf.fit(dataset)
    return clf.predict(dataset) == -1


# Load the data
data = pd.read_csv('data/data.csv')


# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Clean the data
data = data.dropna()  # Remove rows with missing values

# Calculate vegetation density
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']


""" # Example of outlier detection using IForest
outliers = detect_outliers_iforest(data['pm25 mcg/m3'].to_frame())
"""


# Prepare the data
X = data.drop(columns=['Geography', 'city'])  # Replace 'Geography' with the name of your target column if needed

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

# Add PCA components to the data
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

# Visualize the PCA results
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    data=data,
    hue='city',
    palette='viridis',  # Change palette to a more vibrant one
    s=100,  # Increase point size
    alpha=0.8  # Add transparency
)
for city in data['city'].unique():
    sns.kdeplot(
        data=data[data['city'] == city],
        x='PCA1',
        y='PCA2',
        alpha=0.5,
        linewidths=1.5
    )

""" #Annotation of the points
for i in range(data.shape[0]):
    plt.text(
        x=data['PCA1'].iloc[i],
        y=data['PCA2'].iloc[i],
        s=str(i),  # Use index or a specific column
        fontsize=8,
        alpha=0.7
    )
"""


plt.title(f'PCA of Dataset\n(Explained Variance: PC1 = {explained_var[0]:.2%}, PC2 = {explained_var[1]:.2%})', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.legend(title='City', fontsize=12, title_fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)  # Add gridlines
plt.tight_layout()
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)