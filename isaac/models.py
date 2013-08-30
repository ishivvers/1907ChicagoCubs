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

    
=======


def InterpolateDailyFlux(args):
    '''
    Provide the integrated daily GEFS shortwave flux (averaged over models
    and interpolated to MESONET locations) for a single day.
    day: index of the day to integrate in f
    hours: np.array of the hours dimension
    array: np.array of all of the data for that day, shape=(11, 5, 9, 16)
    lon: np.array of the longitude dimension
    lat: np.array of the latitude dimension
    mesonet_locs: a np.recfromcsv record of the mesonet locations
    '''
    day, hours, array, lon, lat, mesonet_locs = args
    print day,
    daily_fluxes = {} # a dictionary of floats recording the daily flux vectors for each station
    for hour in xrange(array.shape[1]):
        ens_fluxes = {} # a dictionary of fluxes from each model for each station at this time
        for ens in xrange(array.shape[0]):
            F = interp2d(lon, lat, array[ens,hour,:,:], kind='linear', bounds_error=True)
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
    x = hours * 3600
    for j,stid in enumerate(daily_fluxes.keys()):
        y = daily_fluxes[stid]
        model_integrated_flux = np.trapz(y,x)
        daily_fluxes[stid] = model_integrated_flux
    return daily_fluxes


def ParallelInterpolatedFlux(f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')):
    '''
    Produce a submission which is simply the integrated (interpolated, model-averaged) flux
    at the locations of the MESONET points.  This is slow, so it is 
    done parallel-wise with the multiprocessing package.
    '''
    days = range( f.variables['time'].shape[0] )
    hours = f.variables['fhour'][:]
    lon = f.variables['lon'][:]-360
    lat = f.variables['lat'][:]
    var = f.variables.keys()[-1]
    arrays = [f.variables[var][:][d] for d in days]
    mesonet_locs = np.recfromcsv('../../data/station_info.csv')
    args = [[d, hours, arrays[d], lon, lat, mesonet_locs] for d in days]
    
    pool = mp.Pool( mp.cpu_count() ) # use all the processors you have!
    submissions = pool.map( InterpolateDailyFlux, args )
    # finally, spit it out as a true submission

    names = [m[0] for m in mesonet_locs]
    predictions = []
    for day in days:
        row = []
        for name in names:
            row.append( submissions[day][name] )
        predictions.append(row)
    save_submission(np.array(predictions), 'InterpFlux_submission.csv')


def SimpleInterpolatedFlux(f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')):
    '''
    Integrate the daily GEFS shortwave flux (averaged over models and 
    interpolated to MESONET locations) and return the result as a submission.
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


if __name__ == '__main__':
    ParallelInterpolatedFlux()
    
>>>>>>> 4663f9e0d916417758e544c69915e36c75db947b
