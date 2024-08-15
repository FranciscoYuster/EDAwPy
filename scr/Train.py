#Importar todas las librerias necesarias 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Leer el archivo CSV para poder tener los datos 
df_train = pd.read_csv('~/onedrive/preciocasas/data/train.csv')

# Mostrar columnas con la idea de identificar las columnas que mayor impacto tienen sobre el precio de la vivienda'
print(df_train.columns)

# Mostramos las variables más correlacionadas
df_num = df_train.select_dtypes(include=['float64', 'int'])
corrmat = df_num.corr()
print(corrmat["SalePrice"].sort_values(ascending=False).head(10))

# Calculamos la matriz de correlacion con las variables de cols transponiendo primero los datos para que cada variable sea una fila
k=10
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=0.9)
# Armamos el mapa de calor
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 11}, yticklabels=cols.values, xticklabels=cols.values)
plt.title('Mapa de calor de SalePrice')
plt.tight_layout()
plt.show()

sns.set()
# Definimos las columnas que mas nos interesaron, y quitamos las que son hermanas
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars',
       'TotalBsmtSF', 'FullBath', 'YearBuilt']
# Armamos nuestra grafica
sns.pairplot(df_train[cols], size=2.5)
plt.title('Correlaciones Precio de Venta')
plt.tight_layout()
plt.show()




print(df_train["SalePrice"].describe())

plt.figure(figsize=(10, 6))  # Tamaño de la figura
sns.histplot(df_train['SalePrice'], kde=True, color='Blue', bins=30, edgecolor="none")  # Ajusta color y número de bins
plt.title('Distribucion de SalePrice')
plt.grid(True)  # Agrega una cuadrícula para una mejor visualización
plt.tight_layout() # Ajustar los màrgenes y el espacio entre las subgràfiacas
plt.show()

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# Guardamos los resultados, y definimos la nueva figura. F y AX es la figura y los ejes 
f, ax = plt.subplots(figsize=(8, 6))
# Definimos FIG como una variable donde haremos un sns.boxplot para guardar la grafica
fig = sns.boxplot(x=var, y="SalePrice", data=data, palette="Set2")
fig.axis(ymin=0, ymax=800000) #Definimos el eje FIG.AXIS
plt.title('Distribucion de SalePrice/OverallQual')
plt.tight_layout()
plt.grid(True)
plt.show()




var = 'GrLivArea' #Se declara la variable que contiene el area vivible
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)# usamos pd.concat para crear una tabla de 2 columnas
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))# Terminamos con un data.plot.scatter para crear la grafica
plt.title('Distribucion de SalePrice/GrLivArea')
plt.grid(True)  # Agrega una cuadrícula para mejor visualización
plt.tight_layout()
plt.show()

var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# Guaradmos los resultados, y definimos la nueva figura. F y AX es la figura y los ejes 
f, ax = plt.subplots(figsize=(8, 6))
# Definimos FIG como una variable donde haremos un sns.boxplot para guardar la grafica
fig = sns.boxplot(x=var, y="SalePrice", data=data, palette="Set2")
fig.axis(ymin=0, ymax=800000) #Definimos el eje FIG.AXIS
plt.title('Distribucion de SalePrice/ GarageCars')
plt.tight_layout()
plt.grid(True)
plt.show()


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.title('Distribucion de SalePrice/TotalBsmtSF')
plt.grid(True)
plt.tight_layout()
plt.show()

var = 'FullBath'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

plt.figure(figsize=(10, 6))
sns.boxplot(x=var, y='SalePrice', data=data)
plt.title('Distribución de SalePrice/FullBath')
plt.grid(True)
plt.tight_layout()
plt.show()


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
palette = sns.color_palette("viridis") 
fig = sns.boxplot(x=var, y="SalePrice", data=data, palette="PuRd")
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)
plt.title('Distribucion de SalePrice/YearBuilt')
plt.tight_layout()
plt.show()





# 4 Datos faltantes
total = df_train.isnull().sum().sort_values(ascending=False) # Idenfiticamos las celdas con valores faltantes en el dataframe con "True", y contamos el nùmero total de valores faltantes.
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) # Dividimos el nùmero de datos faltantes entre el nùmero total de entradas. Esto da como resultado el porcentaje de datos faltantes
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # Unimos las dos series en un dataframe, alineando las columnas, y le asignamos los nombres.
print(missing_data.head(20)) #Mostramos

# Eliminando columnas con más de 1 dato faltante
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, axis=1)

# Eliminando filas con datos faltantes en 'Electrical'
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

# Verificando que no haya datos faltantes
dfnul = df_train.isnull().sum().max()  # Solo verificando que no haya datos faltantes...
print(dfnul)





# Convertir la columna 'SalePrice' a un array de numpy y luego reestructurarla para que sea un array 2D
saleprice_array = df_train['SalePrice'].values.reshape(-1, 1)

# Estandarizar 'SalePrice'
saleprice_scaled = StandardScaler().fit_transform(saleprice_array) # Convierte la serie SalePrice en una matriz de 2 dimensiones, que se necesita para fit_transform

# Ordenar los valores escalados y seleccionar los 10 valores más bajos y más altos
#saleprice_scaled[:, 0].argsort(): Ordena los valores estandarizados de SalePrice en orden ascendente y argsort() devuelve los índices que ordenarían una matriz
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10] 
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]

print('Rango bajo de la distribución:')
print(low_range)
print('\nRango alto de la distribución:')
print(high_range)





var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.tight_layout()
plt.show()

print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2])

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#--------

var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
plt.tight_layout()
plt.show()


var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.tight_layout()
plt.show()








#5 MODO SERIO

sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure() 
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.tight_layout()
plt.show()



# Transformación logaritmica 
df_train['SalePrice'] = np.log(df_train['SalePrice'])
# Transformamos a un gráfio de probabilidad
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.tight_layout()
plt.show()



# Repetimos
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.tight_layout()
plt.show()



df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
plt.tight_layout()
plt.show()





sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
plt.tight_layout()
plt.show()


df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.tight_layout()
plt.show()




#"homeostacidad"

#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
plt.title("Precio de Venta Transformado")
plt.tight_layout()
plt.show()




plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
plt.title("Área Total del sótano Transformado")
plt.tight_layout()
plt.show()



#convertir las variables categóricas en ficticias
df_train = pd.get_dummies(df_train)


#conclusiòn