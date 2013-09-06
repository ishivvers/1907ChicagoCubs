from datalib import *
#import sep_utility
from sklearn.ensemble import RandomForestRegressor
from sklearn import grid_search
import numpy as np
try:
    import multiprocessing as mp
except:
    pass

def evaluate( model, data ):
    """
    A function to get mean absolute error in order to evaluate the model
    s and e are the number of sites and events (days)
    """
    
    s,e = data.shape
    abs_dif = abs(data - model)
    return np.sum( abs_dif)/(s*e)

def random_split( trainX, trainY):
    """
    A function that creates random training and test sets out of the training data.
    """

    l = trainX.shape[0]
    ll = round(l/10)
    permute = np.random.permutation( range(l))
    train_setX, train_setY = trainX[permute[ll:]], trainY[permute[ll:]]
    test_setX, test_setY = trainX[permute[:ll]], trainY[permute[:ll]]
    return train_setX, train_setY, test_setX, test_setY

def RunRandomForest( trainX, trainY, testX ):
    """
    This runs a Random Forest, first fitting two of the hyperparameters via
    cross-validation. Hyperparameter test ranges will need to be adjusted as
    number of parameters changes.
    """
    
    try:
        n_jobs=int(.75*mp.cpu_count())
    except: n_jobs = 1
    n_vals = range(100,800,50)
    parameters = {'n_estimators':n_vals, 'max_features':n_vals}
    model = RandomForestRegressor()
    clf = grid_search.GridSearchCV( model, parameters, n_jobs = n_jobs )
    clf.fit( trainX, trainY )                  
    print clf.grid_scores_
    print clf.best_estimator_
    fit_model = clf.best_estimator_
    #fit_model = RandomForestRegressor(n_estimators = 600, max_features = 600)
    #fit_model.fit( trainX, trainY)
    predictions = fit_model.predict( testX )
    return fit_model, predictions


F = features( which = 'train', verbose = True )
F.calc_all_features()
trainX_full = F.features
times, trainY_full = load_MESONET('train.csv')

trainX_full, trainY_full = trainX_full[:500], trainY_full[:500]
trainX, trainY, testX, testY = random_split( trainX_full, trainY_full)

RF_model, RF_prediction = RunRandomForest( trainX, trainY, testX)
RF_score = evaluate( RF_prediction, testY )

print RF_score
