import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
from LinearRegression import r2_score


dataset = pd.read_excel(r"D:\Universita\IV-anno\ML e DL\Esercizi\Datasets\Combined_cycle_plants.ods", engine='odf', sheet_name="Sheet2")
dataset.head()

target = dataset['PE']
y = np.asarray(target)
X = dataset.iloc[:, 0:4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3,
                                                    stratify=None, #preserve target proportions 
                                                    random_state= 123) #fix random seed for replicability

print(X_train.shape, X_test.shape)

regressor = LinearRegression( learning_rate=0.1, mode=2, )
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

r2_score(y_test, y_pred)