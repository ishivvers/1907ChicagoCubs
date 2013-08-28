'''
A library of models and model helper functions for the 
2013 AMS Solar Energy project.
'''

from datalib import *
from sklearn import metrics
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


'''
# example: fit a model to a few of the variables (each variable averaged across models and hours)
trainX = load_GEFS('train', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
times,trainY = load_MESONET('train.csv')
testX = load_GEFS('test', variables=['dlwrf_sfc','dswrf_sfc','tcdc_eatm','tmp_2m'])
model = RunLassoCV( trainX, trainY, testX )
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
    
    