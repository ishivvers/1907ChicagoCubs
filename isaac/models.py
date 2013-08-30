'''
A library of models and model helper functions for the 
2013 AMS Solar Energy project.
'''

from datalib import *
from sklearn import linear_model, grid_search
from multiprocessing import cpu_count


'''
################################################################################################
# example: fit a model to a few of the variables (each variable averaged across models and hours)

trainX = load_GEFS('train', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
times,trainY = load_MESONET('train.csv')
testX = load_GEFS('test', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
model = RunLassoCV( trainX, trainY, testX )
# display the importance of various variables
plt.figure(figsize=(14,4))
plt.imshow(model.coef_)
plt.xlabel('feature')
plt.ylabel('model')
plt.show()

'''
def RunLassoCV( trainX, trainY, testX, verbose=True, save=True ):
    # use the built-in cross-validation routine to 
    #  figure out the best alpha parameter (alpha determines
    #  how sparse the coefficients are).
    if verbose: print '\nChoosing best alpha and fitting model','#'*20,'\n'
    model = linear_model.LassoCV(normalize=True, verbose=verbose)
    model.fit( trainX, trainY )
    
    # now produce the output estimates
    if verbose: print '\nProducing estimates','#'*20,'\n'
    predictions = model.predict( testX )
    if save: save_submission( predictions, 'LassoCV_submission.csv' )
    
    if verbose: print '\nComplete.'
    
    return model

'''
trainX = load_GEFS('train', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
times,trainY = load_MESONET('train.csv')
testX = load_GEFS('test', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
model = RunLassoCVParallel( trainX, trainY, testX, np.logspace(-1,4,20) )
# display the importance of various variables
plt.figure(figsize=(14,4))
plt.imshow(model.coef_)
plt.xlabel('feature')
plt.ylabel('model')
plt.show()
'''
def RunLassoCVParallel( trainX, trainY, testX, alphas, verbose=True, save=True, n_jobs=int(.75*cpu_count()) ):
    '''
    Run a || grid search for optimal parameters.
    '''
    if verbose: print '\nChoosing best alpha and fitting model','#'*20,'\n'
    model = linear_model.Lasso(normalize=True)
    cvs = grid_search.GridSearchCV(model, {'alpha':alphas}, n_jobs=n_jobs, verbose=int(verbose))
    cvs.fit( trainX, trainY )
    model = cvs.best_estimator_
    
    if verbose: print '\nProducing estimates','#'*20,'\n'
    predictions = model.predict( testX )
    if save: save_submission( predictions, 'LassoCV_submission.csv' )
    
    if verbose: print '\nComplete.'
    
    return model

    
