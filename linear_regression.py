import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data.csv')
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']

X = data['Veg density'].values.reshape(-1, 1)  # Predictor
y = data['pm25 mcg/m3'].values.reshape(-1, 1)  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)

print(f"MSE: ", MSE)
print("Coefficients:", model.coef_)



plt.scatter(X_test, y_test, color='blue', label='Actual values')
plt.plot(X_test, y_pred, color='red', label='Predicted values')
plt.xlabel('Area (km2)')
plt.ylabel('pm25 (µg/m³)')
plt.title('Linear Regression Model')
plt.figtext(0.2,0.80,"MSE: "+str(MSE))
plt.figtext(0.2,0.76,"Coefficients: "+str(model.coef_))
plt.legend()
plt.show()

