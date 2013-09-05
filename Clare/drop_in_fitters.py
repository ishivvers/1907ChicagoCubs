from datalib import *
import sep_utility
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search
import numpy as np

def random_split( trainX, trainY):

    l = trainX.shape[0]
    ll = round(l/10)
    permute = np.random.permutation( range(l))
    train_setX, train_setY = trainX[permute[ll:]], trainY[permute[ll:]]
    test_setX, test_setY = trainX[permute[:ll]], trainY[permute[:ll]]
    return train_setX, train_setY, test_setX, test_setY

def RunRandomForest( trainX, trainY, testX ):
    n_vals = range(10,20,2)
    parameters = {'n_estimators':n_vals, 'max_features':n_vals}
    model = RandomForestRegressor()
    clf = grid_search.GridSearchCV( model, parameters )
    clf.fit( trainX, trainY )                  
    print clf.grid_scores_
    print clf.best_estimator_
    fit_model = clf.best_estimator_

    predictions = fit_model.predict( testX )
    return fit_model, predictions

F = features( which = 'train', verbose = True )
F.calc_all_features()
trainX_full = F.features
times, trainY_full = load_MESONET('train.csv')

trainX_full, trainY_full = trainX_full[:2000], trainY_full[:2000]
trainX, trainY, testX, testY = random_split( trainX_full, trainY_full)

RF_model, RF_prediction = RunRandomForest( trainX, trainY, testX)
RF_score = sep_utility.evaluate( RF_prediction, testY )

print RF_score
