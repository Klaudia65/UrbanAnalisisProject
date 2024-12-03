import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

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

# Add PCA components to the data
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

encoder = OneHotEncoder()
city_encoded = encoder.fit_transform(data[['city']]).toarray()
encoded_df = pd.DataFrame(encoded_city, columns=encoder.get_feature_names_out())
data = pd.concat([data, encoded_df], axis=1)
data.drop('city', axis=1, inplace=True)

# Visualize the PCA results
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PCA1', y='PCA2', data=data, hue='city') 
plt.title('PCA of Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)