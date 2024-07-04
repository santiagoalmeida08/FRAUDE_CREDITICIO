#Paquetes necesarios 
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder #Codificacion de variables
from sklearn.preprocessing import StandardScaler #Escalado de variables
from sklearn.model_selection import train_test_split #Division de datos para entrenar modelos 
from sklearn.model_selection import cross_val_score,cross_validate #Validacion cruzada para entrenamiento de modelos 

#Loading balanced data

df = pd.read_parquet('data//creditcard_balanced.parquet')