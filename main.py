import pandas as pd

# Load the data
data = pd.read_csv('data/all_data.csv')

# Explore the data
print(data.head())
print(data.info())
print(data.describe())

# Clean the data
data = data.dropna()  # Remove rows with missing values

# Analyze the data
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data)
plt.show()

# Prepare the data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop('target_column', axis=1)  # Replace 'target_column' with the name of your target column
y = data['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)