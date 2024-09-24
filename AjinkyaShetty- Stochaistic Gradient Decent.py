"""
Author: Ajinkya Shetty as8856
Simulation of Linear Regression with Stochastic Gradient Descent Method
"""

import numpy as np
import matplotlib.pyplot as plt

#Generate synthetic data
np.random.seed(42)

# X features for prediction of car prices
age = 20 * np.random.rand(100, 1)  # Car age between 0 and 20 years
miles = 150 * np.random.rand(100, 1)  # Miles travelled between 0 and 150k miles
hp = 300 * np.random.rand(100, 1) + 100  # Horsepower between 100 and 400 hp

# Feature matrix
X = np.c_[age, miles, hp]

# Standardization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

# Target variable
y = 20 - 1 * age - 0.5 * miles + 0.3 * hp + np.random.randn(100, 1)  # Car prices in 1000$

# Adding bias to feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Checking the relationship/correlation between feature and target variable
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(age, y, color='blue')
axs[0].set_xlabel('Car Age (years)')
axs[0].set_ylabel('Car Price ($1000)')
axs[0].set_title('Car Age vs Price')

# Checking the relationship/correlation between feature and target variable
axs[1].scatter(miles, y, color='green')
axs[1].set_xlabel('Miles Travelled (1000 miles)')
axs[1].set_ylabel('Car Price ($1000)')
axs[1].set_title('Mileage vs Price')

# Checking the relationship/correlation between feature and target variable
axs[2].scatter(hp, y, color='red')
axs[2].set_xlabel('Horsepower (hp)')
axs[2].set_ylabel('Car Price ($1000)')
axs[2].set_title('Horsepower vs Price')

plt.show()


#calculate mse
def compute_mse(X_b, y, theta):
    predictions = X_b.dot(theta)
    mse = np.mean((predictions - y) ** 2)
    return mse


# Set learning rate and number of iterations
learning_rate = 0.1
n_iterations = 100
batch_size = 5  # Stochastic Gradient Descent for batch size 3

# Initialize parameters randomly
theta = np.zeros((X_b.shape[1], 1))

# Array to store the mse values after every iteration
mse_values = []

# Stochastic Gradient Descent
for iteration in range(n_iterations):
    random_index = np.random.randint(len(X))
    xi = X_b[random_index:random_index + batch_size]
    yi = y[random_index:random_index + batch_size]
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    theta = theta - learning_rate * gradients

    mse = compute_mse(X_b, y, theta) # Calculate mse value
    print("Iteration:",iteration)
    print("MSE:",mse)
    mse_values.append(mse) # Add mse value to the array

print("Final MSE:", mse_values[-1])

# MSE change plot to show the change in the values from the iterations
plt.plot(range(n_iterations), mse_values, label='MSE during Training')
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('MSE Change after every iteration')
plt.legend()
plt.show()

# To predict the y values after training is complete
y_pred = X_b.dot(theta)

# Actual Value vs Predicated Values
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y)), y, color='blue', label='Actual Values')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)
plt.xlabel('Sample Data')
plt.ylabel('Car Price ($1000)')
plt.title('Actual vs Predicted Car Prices')
plt.legend()
plt.show()
