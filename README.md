# Stochastic-Gradient-Descent
Simulation of Linear Regression with (Stochastic) Gradient Descent Method

1) Data Simulation:
In this part, you will simulate your own dataset for linear regression. Follow these steps:
a). Define the problem: Choose a specific problem or scenario for which linear regression is applicable.
For example, you can simulate data related to predicting house prices based on features like size, number
of bedrooms, and location.
b). Generate features: Create a set of feature vectors (X) with multiple features.
You can use random values or a specific pattern to generate these features.
c). Create the target variable: Generate a target variable (Y) that depends on the features. You can introduce
noise to make the problem more realistic.

2) Visualize the data :
Plot the features against the target variable to visualize the relationship between them.

![Figure_1](https://github.com/user-attachments/assets/e1e3c3a0-1477-45e8-aaf4-04f5620aacfe)

4) Implementing Linear Regression with SGD 
In this part, you will write Python code to implement linear regression using the stochastic gradient
descent method. Your script should include the following:
a). Loading the simulated dataset: Use the features (X) and target variable (Y) generated in Part 2.
b). Implementing the linear regression model: Write Python functions to perform the following: Initialize model parameters (coefficients)
randomly.

![Figure_2](https://github.com/user-attachments/assets/f770ce18-0cd5-49eb-8556-bc7df75e70d6)

Implement the stochastic gradient descent algorithm to update the coefficients.
Track the cost function (e.g., Mean Squared Error) during training.
Training the model: Train the linear regression model using SGD on the training set for a specified number of epochs.
Choose an appropriate learning rate and batch size.
Evaluating the model: Calculate and print the Mean Squared Error (MSE) to assess the model’s performance.

![Figure_3](https://github.com/user-attachments/assets/023eb7ba-1da0-4ffd-bf4a-441d594e668f)

4) Analysis and Discussion:
In this section, briefly analyze and discuss the results and the implementation: Report the values of
MSE obtained from your simulation. What do these metrics tell you about the model’s performance
on the simulated data?

