from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris=load_iris()

X = iris.data
y = iris.target

#mencoba akurasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
print('tingkat akurasinya : {}'.format(knn.score(X_test, y_test)))

#prediksi baru
y_prediksi=knn.predict(X_test)
X_baru=np.array([2,5,2.5,2], ndmin=2)
prediksi=knn.predict(X_baru)
print("prediksinya adalah : {}".format(prediksi))