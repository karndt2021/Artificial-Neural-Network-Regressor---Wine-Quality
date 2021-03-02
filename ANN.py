"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_epochs = 100
csv_file = 'winequality-white.csv'
num_cols = 11
num_cols_scale = 11
layer1_nodes = 150
layer2_nodes = 150
act_function = 'relu'
optimizer = 'adam'
set_verbose = 2
num_folds = 10
batch_size = 300

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataframe = pd.read_csv(csv_file, delimiter = ";", header = None)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataframe = dataframe.replace(np.nan, 0)
dataset = dataframe.values

X = dataset[:,0:num_cols]
Y = dataset[:,num_cols]

X_MinMax = preprocessing.MinMaxScaler()
Y_MinMax = preprocessing.MinMaxScaler()
Y=np.array(Y).reshape(len(Y),1)
X = X_MinMax.fit_transform(X)
Y = Y_MinMax.fit_transform(Y)
print(X_MinMax.scale_)
print(Y_MinMax.scale_)

print(X.shape)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def build_model():
     model = Sequential()
     model.add(Dense(layer1_nodes, activation= act_function, input_shape=(X.shape[1],)))
     model.add(Dense(layer2_nodes, activation= act_function))
     model.add(Dense(layer2_nodes, activation= act_function))
     model.add(Dense(1))
     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
     return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
K-fold validation
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_val = np.concatenate((X, X_test), axis = 0)
Y_val = np.concatenate((Y, Y_test), axis = 0)

acc_per_fold = []
loss_per_fold = []

# Define K-Fold
kfold = KFold(n_splits = num_folds, shuffle = True)

fold_no = 1
for train, test in kfold.split(X_val,Y_val):
    model = build_model()
    print(f'Training for fold {fold_no} ...')
    
    history = model.fit(X_val[train], Y_val[train], batch_size=batch_size,
                        epochs = num_epochs, verbose = set_verbose)
    scores = model.evaluate(X_val[test], Y_val[test], verbose = 0)
    print(f'Fold {fold_no} score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}')
    acc_per_fold.append(scores[1]*100)
    loss_per_fold.append(scores[0])
    
    fold_no = fold_no + 1



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show & Plot output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(np.mean(loss_per_fold)) 
print(min(loss_per_fold))
print('Minimum loss at fold:', loss_per_fold.index(min(loss_per_fold)))
plt.plot(range(1, len(loss_per_fold) + 1), loss_per_fold)
plt.xlabel('Folds')
plt.ylabel('Validation MSE')
axes = plt.gca()
axes.set_ylim([0.0100,0.0180])
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Final Model Application
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()
model = build_model()
test_mse_score = []
estimator = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, verbose=set_verbose)
test_mse_score, test_mae_score = model.evaluate(X_test, Y_test)
print('test MSE: ',test_mse_score) 
print('Mean Val MSE: ',np.mean(loss_per_fold)) 
stop_time = datetime.datetime.now()
print("Time:", stop_time - start_time)
plt.plot(estimator.history['loss'])
plt.plot(estimator.history['mae'])
plt.show()
