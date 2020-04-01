import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Trae el dataset
dataset=pd.read_csv('datasetsalarios.csv')

#imprime las primeras 5 lineas del dataset
print(dataset.head(5))

#Imprime las dimensiones del dataset
print(dataset.shape)
#Separa los datos por columnas

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

X_train,X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print(regressor.score(X_test,Y_test))


viz_train =plt
#Scatter crea una seccion punteada dentro del plano
viz_train.scatter(X_train, Y_train, color='blue')

#plot grafica la recta
viz_train.plot(X_train,regressor.predict(X_train),color='black')
viz_train.title('Salario vs Experiencia')
viz_train.xlabel('Experiencia')
viz_train.ylabel('Salario')
viz_train.show()
