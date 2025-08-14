import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {'exp' : [5,6,7,8,9,10,11,12,13,14],
        'salary' : [23000,24000,25000,26000,27000,28000,29000,30000,31000,32000]}

df = pd.DataFrame(data)

x = df[['exp']]
y = df[['salary']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print("X-Train" , x_train)
print("X-Test" , x_test)
print("Y-Train" , y_train)
print("Y-Test" , y_test)
      
#df.to_csv('User.csv')
print(df)
print("---------------------------------------------------------------")
print(df.head(1))
print("---------------------------------------------------------------")
print(df.describe())
df = pd.read_csv(r'C:\OMEGA\Machine Learning\MachineLearningLab\User.csv')
print(df)
print("---------------------------------------------------------------")

# Applying Linear Regession Model
model  = LinearRegression()
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print("Predicted Values: ", y_prediction)
print("---------------------------------------------------------------")

# Model Accuracy
print("Mean Squared Error: ", mean_squared_error(y_test, y_prediction))
print("Accuracy : ", r2_score(y_test, y_prediction))

# Plotting the results
plt.scatter(y_test, y_prediction, color='red')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.show()