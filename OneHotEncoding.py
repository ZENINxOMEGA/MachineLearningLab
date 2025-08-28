from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

data = {
    'Age': [27, 28, 29, None, 30, 32, 34, None, 35, 38],
    'Height': [160, 165, 168, None, 169, 170, None, 180, 189, 200],
    'Weight': [54, 56, 68, 69, 70, 71, None, 72, 73, 74],
    'City': ["Delhi", "Kolkata", np.nan, "Kolkata", "Gujarat", np.nan, "China", "Bihar", "Kanpur", "Uttar Pradesh"]
}
df = pd.DataFrame(data)
print(df)
print("---------------------------------------------------------------------------------------------------------------")
print()

ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[['City']])
ohe_columns = ohe.get_feature_names_out(['City'])
df_ohe = pd.DataFrame(ohe_array, columns=ohe_columns)
print(df_ohe)

le = LabelEncoder()
df['le_City'] = le.fit_transform(df['City'])
print(df)

print("---------------------------------------------------------------------------------------------------------------")