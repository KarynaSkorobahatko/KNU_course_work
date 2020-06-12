import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import ensemble

data = pd.read_csv('2019_accidents.csv')
print(data.head())
print(data.info())

corr_data = data.corr()

# del data['District Name']
del data['Descripcio_torn']
# del data['Numero_lesionats_lleus']
del data['Numero_expedient']
del data['NK_Any']
del data['Descripcio_dia_setmana']

print(data.info())

# Графік кореляції
# plt.figure(figsize=(16,16))
# sns.heatmap(data.corr(), annot=True)
# plt.show()

sb.scatterplot(data['Numero_vehicles_implicats'], data['Numero_victimes'])

# my_X = df.drop(['Vehicles involved'], axis=1)
# my_Y = df['Vehicles involved'].values.reshape(-1,1)
X = data[['Hora_dia', 'Mes_any', 'Numero_lesionats_lleus', 'Numero_victimes', 'Numero_lesionats_greus', 'Longitud',
          'Latitud']]
# y = data['Numero_lesionats_greus'].values.reshape(-1, 1)
y = data['Numero_vehicles_implicats']
cer_s = scale(X)
y = y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=25)

N_train, _ = X_train.shape
N_test, _ = X_test.shape
print(N_train, N_test)

# kNN – метод ближайших соседей¶
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

kn_train = np.mean(y_train != y_train_predict)
kn_test = np.mean(y_test != y_test_predict)
print('KNeighborsClassifier without GridSearchCV: ', 'real = ', kn_train, "test =", kn_test)

# подбор параметров
n_neighbors_array = [15, 20, 25, 30, 35, 40, 45, 50, 55]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print('GridSearchCV for test data:', best_cv_err, 'best count of neighbors =', best_n_neighbors)

knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

knn_train = np.mean(y_train != knn.predict(X_train))
knn_test = np.mean(y_test != knn.predict(X_test))
print('KNeighborsClassifier with GridSearchCV: ', 'real = ', knn_train, "test =", knn_test)

# SVC – машина опорных векторов
svc = SVC()
svc.fit(X_train, y_train)

svc_train = np.mean(y_train != svc.predict(X_train))
svc_test = np.mean(y_test != svc.predict(X_test))
print('SVC: ', 'real = ', svc_train, "test =", svc_test)

# ліс
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test = np.mean(y_test != rf.predict(X_test))
print('RandomForestClassifier: ', 'real = ', err_train, "test =", err_test)
