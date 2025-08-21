# Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample Data Creation
# Creating a dictionary with sample data including missing values (None and np.nan)
data = {
    'Age': [27, 28, 29, None, 30, 32, 34, None, 35, 38],
    'Height': [160, 165, 168, None, 169, 170, None, 180, 189, 200],
    'Weight': [54, 56, 68, 69, 70, 71, None, 72, 73, 74],
    'City': ["Delhi", "Kolkata", np.nan, "Kolkata", "Gujarat", np.nan, "China", "Bihar", "Kanpur", "Uttar Pradesh"]
}

# Converting dictionary to DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("---------------------------------------------------------------")

# Analyzing Missing Values
# Check and display the count of missing values in each column
print("Missing Values in Each Column:")
print(df.isnull().sum())
print("----------------------------------------------------------------")

# Method 1: Handling Missing Values - Complete Case Analysis
# Dropping all rows with any missing values (not recommended if data loss is significant)
df_drop = df.dropna()
print("DataFrame after dropping rows with missing values:")
print(df_drop)
print("---------------------------------------------------------------")

# Method 2: Handling Missing Values - Imputation
# Age: Using mean imputation for numerical data
df['Age'].fillna(df['Age'].mean(), inplace=True)
# Height: Using median imputation for numerical data
df['Height'].fillna(df['Height'].median(), inplace=True)
# Weight: Using forward fill (carrying forward the last valid observation)
df['Weight'].fillna(method='ffill', inplace=True)
# City: Using mode imputation for categorical data
df['City'].fillna(df['City'].mode()[0], inplace=True)

print("DataFrame after handling missing values:")
print(df)
print("---------------------------------------------------------------")

# Data Transformation
# 1. Min-Max Normalization (scales data to a fixed range, typically [0, 1])
mm = MinMaxScaler()
df[['mm_Age', 'mm_Height']] = mm.fit_transform(df[['Age', 'Height']])
print("DataFrame after Min-Max Normalization:")
print(df)
print("---------------------------------------------------------------")

# 2. Standardization (scales data to have mean=0 and variance=1)
std = StandardScaler()
df[['std_Age', 'std_Height']] = std.fit_transform(df[['Age', 'Height']])
print("DataFrame after Standardization:")
print(df)
print("---------------------------------------------------------------")   