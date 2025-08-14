import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Task - 1
data = {'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        'Sales': [200, 220, 250, 270, 300, 320, 350, 400, 450, 500]}

df = pd.DataFrame(data)
#df.to_csv('Sales.csv')
print(df)

# Task - 2

df = pd.read_csv('Sales.csv')
print("\nFirst 5 rows of Sales.csv:")
print(df.head())

# Task - 3

plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['Sales'], color='red')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales Distribution Over Years')
plt.grid(True)
plt.show()

# Task - 4
# Prepare the data for linear regression
X = df[['Year']].values  # Features (independent variable)
y = df['Sales'].values   # Target (dependent variable)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions for year 2026
future_year = [[2026]]
predicted_sales = model.predict(future_year)

print("\nModel Information:")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"\nPredicted Sales for year 2026: ${predicted_sales[0]:.2f}")

# Visualize the regression line along with scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')

# Add the future prediction point
plt.scatter(future_year, predicted_sales, color='green', marker='*', s=200, label='2026 Prediction')

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Sales Trend with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Task - 5
# Split the data into training and testing sets (80-20 split)
X = df[['Year']].values
y = df['Sales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")

