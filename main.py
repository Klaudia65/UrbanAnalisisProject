import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/all_data.csv')
data['city'] = 'NYC' 
data.loc[43:, 'city'] = 'S'
print(data)
print(data.describe())

# Clean the data
data = data.dropna()  # Remove rows with missing values

print(data.shape)