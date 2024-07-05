#Paquetes necesarios 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder #Codificacion de variables
from sklearn.preprocessing import StandardScaler #Escalado de variables
from sklearn.model_selection import train_test_split #Division de datos para entrenar modelos 
from sklearn.model_selection import cross_val_score,cross_validate #Validacion cruzada para entrenamiento de modelos 
from sklearn.feature_selection import SelectFromModel #Seleccion de variables
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #Modelos a utilizar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score#Metricas para evaluar modelos
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay #Metricas para evaluar modelos

#Loading balanced data

df = pd.read_parquet('data//creditcard_balanced.parquet')
df.dtypes

#Division de datos

x = df.drop('Class',axis = 1)
y = df['Class']

#Encoding data --- no aplica porque todas las variables son numericas

#Escalado de variables
scaler = StandardScaler()
x_esc = scaler.fit_transform(x)

#Modelos a utilizar 
rf = RandomForestClassifier()
lr = LogisticRegression()

modelo = rf

#Train_test_split model
x_train,x_test,y_train,y_test = train_test_split(x_esc,y,test_size=0.3,random_state=42)
modelo.fit(x_train,y_train)
y_pred = modelo.predict(x_test)

print(classification_report(y_test,y_pred))

#Metricas
print(f'Acuraccy_train : {modelo.score(x_train,y_train)}')

print(f'Accuracy : {accuracy_score(y_test,y_pred)}')
print(f'Precision : {precision_score(y_test,y_pred)}')
print(f'Recall : {recall_score(y_test,y_pred)}')
print(f'F1 : {f1_score(y_test,y_pred)}')
print(f'ROC_AUC : {roc_auc_score(y_test,y_pred)}')

#Matriz de confusion
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


#Cross_validtaion model 
cv = cross_val_score(modelo,x_esc,y,scoring='f1',cv=10)
cv.mean()

#Cross validate evaluation

cross_eval = cross_validate(modelo,x_esc,y,scoring='f1',cv=10,return_train_score=True)

print(f'F1_train : {cross_eval["train_score"].mean()}')
print(f'F1_test : {cross_eval["test_score"].mean()}')