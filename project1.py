import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load Dataset
data = pd.read_csv("C:\\Users\\Wick\\Documents\\Book1.csv")

# Features (Hours) and Target (Score)
x = data[['Hours']]
y = data[['Score']]

# Split data into Training and Testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Create and Train Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on Test Data
predicted_scores = model.predict(x_test)

# Model Evaluation
mae = mean_absolute_error(y_test, predicted_scores)
mse = mean_squared_error(y_test, predicted_scores)
rmse = np.sqrt(mse)

# Display Evaluation Results
print("===== MODEL EVALUATION =====")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Predict Score for 7 Hours
new_prediction = model.predict([[7]])

print("\n===== NEW PREDICTION =====")
print(f"Predicted Score for 7 Hours of Study: {new_prediction[0][0]:.2f}")

# Model Equation
print("\n===== MODEL DETAILS =====")
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# Visualization
plt.figure(figsize=(8, 5))

# Scatter plot of actual data
plt.scatter(x, y, label="Actual Data")

# Regression line
plt.plot(x, model.predict(x), linewidth=2, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction Using Linear Regression")
plt.legend()
plt.grid(True)

plt.show()
