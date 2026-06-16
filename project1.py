import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error
import numpy as np

data = pd.read_csv("C:\\Users\\Wick\\Documents\\Book1.csv")

x = data[['Hours']]    # double brackets for 2d input
y = data[['Score']]  # Target column 

model = LinearRegression()
model.fit(x,y)

predicted_score = model.predict(x)

# model evalaution 
mae =  mean_absolute_error(y, predicted_score)
mse = mean_squared_error( y,predicted_score)
rmse = np.sqrt(mse)

# show results
print("Mean Absolute ERROR (MAE): " , mae)
print("Mean Squared ERROR (MSE): " , mse)
print("Root Mean Squared ERROR (RMSE): " , rmse)


new_prediction = model.predict([[7]])
print(f"Prediction Score for 7 hour is score = {new_prediction}")