#En este script se hace la conversion de un archivo CSV a un archivo Parquet 
#Por cuestion de almacenamiento del git

import pandas as pd

#Loading-data
df = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Proyectos_DC\\creditcard.csv',sep=',')

#Tranform and save data

df.to_parquet('C:\\Users\\Usuario\\Desktop\\Proyectos_DC\\FRAUDE_CREDITICIO\\data\\creditcard.parquet',engine='fastparquet') 