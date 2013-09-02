'''
A library of models and model helper functions for the 
2013 AMS Solar Energy project.
'''

from datalib import *
from sklearn import linear_model, grid_search
import multiprocessing as mp


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
trainX = load_GEFS('train', variables=['dlwrf_sfc','dswrf_sfc','tmp_2m'])
times,trainY = load_MESONET('train.csv')
testX = load_GEFS('test', variables=['dlwrf_sfc','dswrf_sfc','tmp_2m'])
model, predictions = RunLassoCVParallel( trainX, trainY, testX, np.logspace(-1,4,20) )
# display the importance of various variables
plt.figure(figsize=(14,4))
plt.imshow(model.coef_)
plt.xlabel('feature')
plt.ylabel('model')
plt.show()
'''
def RunLassoCVParallel( trainX, trainY, testX, alphas, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
    '''
    Run a || grid search for optimal alpha parameter in a Lasso model.
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
    
    return model, predictions

    
def RunElasticNetCVParallel( trainX, trainY, testX, params, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
    '''
    Run a || grid search for optimal parameters in an Elastic Net model.
    '''
    if verbose: print '\nChoosing best parameters and fitting model','#'*20,'\n'
    model = linear_model.ElasticNet(normalize=True)
    cvs = grid_search.GridSearchCV(model, params, n_jobs=n_jobs, verbose=int(verbose))
    cvs.fit( trainX, trainY )
    model = cvs.best_estimator_

    if verbose: print '\nProducing estimates','#'*20,'\n'
    predictions = model.predict( testX )
    if save: save_submission( predictions, 'ElasticNetCV_submission.csv' )

    if verbose: print '\nComplete.'

    return model, predictions
    
def SimpleInterpolatedFlux(f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')):
    '''
    Integrate the daily GEFS shortwave flux (averaged over models and 
    interpolated to MESONET locations) and return the result as a submission.
    This is slow.
    '''
    var = f.variables.keys()[-1]
    mesonet_locs = np.recfromcsv('../../data/station_info.csv')
    
    submissions = {}
    print 'calculating errors:',f.variables[var].shape[0],'total'
    for i,day in enumerate(xrange(f.variables[var].shape[0])):
        print i
        daily_fluxes = {} # a dictionary of lists recording the daily flux vectors for each station
        for hour in xrange(f.variables[var].shape[2]):
            ens_fluxes = {} # a dictionary of fluxes from each model for each station at this time
            for ens in xrange(f.variables[var].shape[1]):
                F = create_2d_interpolation(f, day, hour, ens)
                for station in mesonet_locs:
                    loc_flux = F( station[2], station[1] )
                    if station[0] in ens_fluxes.keys():
                        ens_fluxes[station[0]].append(loc_flux)
                    else:
                        ens_fluxes[station[0]] = [loc_flux]
            # average over models, and append to the daily flux vector dictionary
            for stid in ens_fluxes.keys():
                ensemble_mean = np.mean(ens_fluxes[stid])
                if stid in daily_fluxes.keys():
                    daily_fluxes[stid].append( ensemble_mean )
                else:
                    daily_fluxes[stid] = [ensemble_mean]
        # integrate the daily flux vector and add to submissions
        #  fluxes are in joules per second per meter squared, so we should
        #  integrate over time in seconds
        for j,stid in enumerate(daily_fluxes.keys()):
            x = f.variables['fhour'][:] * 3600
            y = daily_fluxes[stid]
            model_integrated_flux = np.trapz(y,x)
            if stid in submissions.keys():
                submissions[stid].append( model_integrated_flux )
            else:
                submissions[stid] = [model_integrated_flux]
                
    # finally, spit it out as a true submission
    names = [m[0] for m in mesonet_locs]
    predictions = np.array([ submissions[stid] for stid in names ]).T
    save_submission(predictions, 'InterpFlux_submission.csv')




######################################################
# Run an ElasticNet model on the calculated features of our data,
# witholding part as a 'test' set.
# Grid search for optimal parameters using the || gridsearch
# function.
######################################################
F = features( which='train', verbose=True )
F.calc_all_features()
times, all_Y = load_MESONET('train.csv')
trainX, trainY, testX, testY = split_train( F.features, all_Y )
params = {'alpha':np.logspace(0,3,25), 'l1_ratio':np.linspace(0,1,10)}
model, predictions = RunElasticNetCVParallel( trainX, trainY, testX, params )
print 'Best model parameters:', model.get_params()
print 'Best model MAE:', MAE( testY, predictions )
