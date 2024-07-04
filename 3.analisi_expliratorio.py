import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE # Balance the data
import joblib

#Load the data

df = pd.read_parquet('data//creditcard.parquet')#El dataset tiene 28 variables que fueron obtenidas tras un proceso de PCA
df.dtypes

#Exploring de data nulls

df.isnull().sum() # No se observan nulos en la base de datos 

#Exploring duplicated data 

df.duplicated().sum() # Se observan 1081 duplicados los cuales no se eliminan ya que son transacciones diferentes que ocurrieron en el mismo lapso tiempo

#Exploring the numerical data , in this data base there aren't categorical data

df.hist(figsize=(20,20)) # The PCA variables are very dificult to interpret.

df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()  # Se observa cierta correlacion entre variables de PCA y la variable objetivo

#Exploring the target variable

df.columns
df['Class'].value_counts() # Se observa un desbalance en la variable objetivo

#Balancing the data

smote = SMOTE(random_state=42)
x_res,y_res = smote.fit_resample(df.drop('Class',axis=1),df['Class'])

#Data after balancing
df_new = pd.concat([x_res,y_res],axis=1)

#Saving the data balanced, taking aleatory 40000 samples

df_sample = df_new.sample(40000)

df_sample.to_parquet('data//creditcard_balanced.parquet',engine='pyarrow')