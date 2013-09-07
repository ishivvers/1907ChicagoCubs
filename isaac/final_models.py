'''
The final library of models and model helper functions for the 
2013 AMS Solar Energy project.
'''

from datalib import *
from sklearn import linear_model, svm, ensemble, grid_search
import multiprocessing as mp
import csv


def RunRandomForest( args, f_psearch=0.1, verbose=True ):
    """
    Run a Random Forest Regressor model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    """
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best parameters on',f_psearch*100,'percent of the data'
    mask = np.random.random( trainX.shape[0] ) < f_psearch
    
    parameters = {'n_estimators':set(np.linspace(5, trainX.shape[1], 50).astype(int)), \
                  'max_features':set(np.linspace(5, trainX.shape[1], 50).astype(int))}
    premodel = ensemble.RandomForestRegressor()
    gridCV = grid_search.GridSearchCV( premodel, parameters, cv=5, verbose=int(verbose) )
    gridCV.fit( trainX[mask], trainY[mask] )                  
    n_estimators = gridCV.best_params['n_estimators']
    max_features = gridCV.best_params['max_features']
    
    if verbose: print '\nUsing params = ',gridCV.best_params_,'\nFitting model on full data'
    model = RandomForestRegressor( n_estimators=n_estimators, max_features=max_features )
    model.fit( trainX, trainY )
    
    if verbose: print '\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions


def RunSVR( args, f_psearch=0.1, verbose=True ):
    '''
    Run an SVR model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best parameters on',f_psearch*100,'percent of the data'
    mask = np.random.random( trainX.shape[0] ) < f_psearch
    
    params = {'C':np.logspace(-5,5,10), 'gamma':np.logspace(-5,5,10), 'epsilon':np.logspace(-2,2,10)}
    premodel = svm.SVR( cache_size=1000 )
    gridCV = grid_search.GridSearchCV( premodel, params, cv=5, verbose=verbose )
    gridCV.fit( trainX[mask], trainY[mask] )
    gamma = gridCV.best_params_['gamma']
    C = gridCV.best_params_['C']
    epsilon = gridCV.best_params_['epsilon']
    
    if verbose: print '\nUsing params = ',gridCV.best_params_,'\nFitting model on full data'
    model = svm.SVR( cache_size=1000, verbose=verbose, C=C, gamma=gamma, epsilon=epsilon )
    model.fit( trainX, trainY )
    
    if verbose: print '\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions


def RunRidge( args, verbose=True ):
    '''
    Run a Ridge model.
    trainX, trainY, testX: you know what those are
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best alpha and fitting the model.'
    model = linear_model.RidgeCV( alphas=np.logspace(-0,3,100), cv=5 )
    model.fit( trainX, trainY )
    
    if verbose: print '\nUsing alpha =',model.alpha_,'\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions



def RunElasticNet( args, verbose=True ):
    '''
    Run an ElasticNet model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best parameters and fitting the model.'
    model = linear_model.ElasticNetCV( l1_ratio=[.5, .7, .9, .95, .99, .999, 1.0], n_alphas=100, cv=5, verbose=int(verbose) )
    model.fit( trainX, trainY )
    
    if verbose: print '\nUsing alpha =',model.alpha_,'and l1_ratio = ',model.l1_ratio_,'\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions



def RunLassoLARS( args, verbose=True ):
    '''
    Run a LassoLARS model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best alpha and fitting the model'
    model = linear_model.LassoLarsCV( cv=5, verbose=int(verbose), normalize=False )
    model.fit( trainX, trainY )
    
    if verbose: print '\nUsing alpha =',model.alpha_,'\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions


def RunLasso( args, verbose=True ):
    '''
    Run a Lasso model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best alpha and fitting the model.'
    model = linear_model.LassoCV( n_alphas=100, cv=5, verbose=int(verbose) )
    model.fit( trainX, trainY )
    
    if verbose: print '\nUsing alpha =',model.alpha_,'\nProducing estimates'
    predictions = model.predict( testX )
    if verbose: print '\nComplete.'
    
    return predictions


def RunStationModels( modelfunc, fname, nproc=-1, n_mesonet=98, verbose=False ):
    '''
    Run an individual instance of modelfunc for each MESONET station parallel-wise,
    using <nproc> cores (nproc=-1 uses all available cores).
    <fname> is the name of the output submission file.
    '''
    if nproc<1 or nproc>mp.cpu_count():
        pool = mp.Pool( mp.cpu_count() )
    else:
        pool = mp.Pool( nproc )
    
    if verbose: print 'Building argument list to distribute amongst cores'
    args_list = []
    times, alltrainY = load_MESONET('train.csv')
    F_train = features( which='train', verbose=verbose )
    F_train.calc_all_features( scale=False )
    F_test = features( which='test', verbose=verbose )
    F_test.calc_all_features( scale=False )
    for i in xrange(n_mesonet):
        # get the training features and the training set scaler
        trainX = F_train.return_feats_near( i, n=9, scale=True )
        trainY = alltrainY[:,i]
        testX = F_test.return_feats_near( i, n=9, scale=False )
        # rescale the test features to the training set scaler
        testX = F_train.scaler.transform( testX )
        args = (trainX, trainY, testX)
        args_list.append(args)
    
    if verbose: print 'Feeding out tasks!'
    predictions_list = pool.map( modelfunc, args_list )
    
    if verbose: print 'Saving result to',fname
    predictions = np.loadtxt( '../../data/sampleSubmission.csv', skiprows=1, delimiter=',' )
    for i in xrange(len(predictions_list)):
        predictions[:, i+1] = predictions_list[i]
    outf = open(fname,'w')
    fwriter = csv.writer( outf )
    for i,row in enumerate(predictions):
        if i == 0:
            # pull in the header from the sampleSubmission
            tmpf = open('../../data/sampleSubmission.csv','r')
            freader = csv.reader( tmpf )
            fwriter.writerow( freader.next() )
            tmpf.close()
        else:
            outrow = row.tolist()
            outrow[0] = int(outrow[0])
            fwriter.writerow( outrow )
    outf.close()
    
    if verbose: print 'All done.'
        

#RunStationModels( RunLasso, 'LassoSubmission.csv', verbose=True )
