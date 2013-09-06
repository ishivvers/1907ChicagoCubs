'''
A library of models and model helper functions for the 
2013 AMS Solar Energy project.
'''

from datalib import *
from sklearn import linear_model, grid_search, feature_selection, svm
import multiprocessing as mp
import csv

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
    

def RunElasticNetCV( trainX, trainY, testX, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
    # use the built-in cross-validation routine to 
    #  figure out the best alpha and rho parameters (alpha determines
    #  how sparse the coefficients are).
    if verbose: print '\nChoosing best parameters and fitting model','#'*20,'\n'
    model = linear_model.ElasticNetCV(normalize=True, verbose=verbose, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=n_jobs)
    model.fit( trainX, trainY )

    # now produce the output estimates
    if verbose: print '\nProducing estimates','#'*20,'\n'
    predictions = model.predict( testX )
    if save: save_submission( predictions, 'ElasticNetCV_submission.csv' )

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

# example:
# params = {'epsilon':np.linspace(0,1,11), 'gamma':np.logspace(-4,4,10), 'C':np.logspace(-2,4,10)}
def RunSVRCVGridSearch( trainX, trainY, testX, params, verbose=True, save=False, n_jobs=int(.75*mp.cpu_count()) ):
    '''
    Run a || grid search for optimal SVR parameters.
    '''
    if verbose: print '\nChoosing best params and fitting model','#'*20,'\n'
    model = svm.SVR(cache_size=1000) #large cache is faster for computers with lotsa ram
    cvs = grid_search.GridSearchCV(model, params, n_jobs=n_jobs, verbose=int(verbose))
    cvs.fit( trainX, trainY )
    model = cvs.best_estimator_
    
    if verbose: print '\nComplete.'
    if verbose: print 'Best parameters:\n', model.get_params()
    
    return model
    

def RunLassoCVGridSearch( trainX, trainY, testX, alphas, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
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

    
def RunElasticNetCVGridSearch( trainX, trainY, testX, params, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
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


def RunRidgeCVParallel( trainX, trainY, testX, alphas, verbose=True, save=True, n_jobs=int(.75*mp.cpu_count()) ):
    '''
    Run a || grid search for optimal parameters in Ridge Regression model.
    '''
    if verbose: print '\nChoosing best parameters and fitting model','#'*20,'\n'
    model = linear_model.Ridge(normalize=True)
    cvs = grid_search.GridSearchCV(model, {'alpha':alphas}, n_jobs=n_jobs, verbose=int(verbose))
    cvs.fit( trainX, trainY )
    model = cvs.best_estimator_

    if verbose: print '\nProducing estimates','#'*20,'\n'
    predictions = model.predict( testX )
    if save: save_submission( predictions, 'RidgeCV_submission.csv' )

    if verbose: print '\nComplete.'

    return model, predictions


def RunLassoFeatureElimination( trainX, trainY, verbose=True, alpha=None ):
    '''
    Use a Lasso fit to figure out what features are important.
    '''
    if alpha == None:
        if verbose: print '\nChoosing an alpha parameter\n','#'*20,'\n'
        model = linear_model.LassoCV()
        model.fit( trainX, trainY )
        best_alpha = model.alpha_
    else:
        best_alpha = alpha
    
    if verbose: print '\nUsing alpha=',best_alpha
    if verbose: print 'Searching for best feature set\n'
    
    model2 = linear_model.Lasso()
    rfe = feature_selection.RFECV( estimator=model2, step=2, verbose=int(verbose) )
    rfe.fit( trainX, trainY )
    
    if verbose: print 'Found reduced feature set'
    return rfe.ranking_

def RunRidgeFeatureElimination( trainX, trainY, verbose=True, alpha=None ):
    '''
    Use a Ridge fit to figure out what features are important.
    '''
    if alpha == None:
        if verbose: print '\nChoosing an alpha parameter\n','#'*20,'\n'
        model = linear_model.RidgeCV( alphas=np.logspace(-1,4,25) )
        model.fit( trainX, trainY )
        best_alpha = model.alpha_
    else:
        best_alpha = alpha

    if verbose: print '\nUsing alpha=',best_alpha
    if verbose: print 'Searching for best feature set\n'

    model2 = linear_model.Ridge()
    rfe = feature_selection.RFECV( estimator=model2, step=0.01, verbose=int(verbose) )
    rfe.fit( trainX, trainY )

    if verbose: print 'Found reduced feature set'
    return rfe
    
    
    
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


def RunLasso( args, f_psearch=0.1, verbose=True ):
    '''
    Run a Lasso model.
    trainX, trainY, testX: you know what those are
    f_psearch: the fraction of the test sample to use to choose hyperparameters (0 < f_psearch < 1)
    '''
    trainX, trainY, testX = args
    if verbose: print '\nChoosing best alpha on',f_psearch*100,'percent of the data'
    mask = np.random.random( trainX.shape[0] ) < f_psearch
    premodel = linear_model.LassoCV( n_alphas=100, cv=5, verbose=int(verbose) )
    premodel.fit( trainX[mask], trainY[mask] )
    alpha = premodel.alpha_
    
    if verbose: print '\nUsing alpha =',alpha,'\nFitting model on full data'
    model = linear_model.Lasso( alpha=alpha )
    model.fit( trainX, trainY )
    
    if verbose: print '\nProducing estimates'
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
    
    if verbose: print 'Saving result as a submission.'
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
        

RunStationModels( RunLasso, 'LassoSubmission.csv', verbose=True )

# 
# ######################################################
# # Run an ElasticNet model on the calculated features of our data,
# # witholding part as a 'test' set.
# # Grid search for optimal parameters using the || gridsearch
# # function.  Takes ~300 minutes on Sirius.
# ######################################################
# F = features( which='train', verbose=True )
# F.calc_all_features()
# times, all_Y = load_MESONET('train.csv')
# trainX, trainY, testX, testY = split_train( F.features, all_Y )
# params = {'alpha':np.logspace(0,3,25), 'l1_ratio':np.linspace(0,1,10)}
# model, predictions = RunElasticNetCVParallel( trainX, trainY, testX, params )
# print 'Best model parameters:', model.get_params()
# print 'Best model MAE:', MAE( testY, predictions )
# # appears to have chosen alpha=1000, l1_ratio=1.0, with MAE=2.3M


# #####################################################
# Run a Ridge to pick features and reduce the feature set.
# Use a random subset of the data, and just a single MESONET location.
# #####################################################
# F = features( which='train', verbose=True )
# F.calc_all_features()
# trainX = F.features
# times, trainY = load_MESONET('train.csv')
# # build a numpy mask with 10% ones and 90% zeros
# mask = []
# mesonet_n = 90
# mesonet_loc = np.recfromcsv('../../data/station_info.csv')[mesonet_n]
# for i in range(trainY.shape[0]):
#     v = np.random.random()
#     if v<=0.2: mask.append(1)
#     else: mask.append(0)
# mask = np.array(mask, dtype='bool')
# rfe = RunRidgeFeatureElimination( trainX[mask], trainY[:,mesonet_n][mask] )
# feat_ranking = rfe.ranking_
# feat_names = np.array( F.featnames )
# good_features = feat_names[ feat_ranking == 1 ]
# print 'good features:',good_features
# plt.scatter( mesonet_loc[2], mesonet_loc[1], s=50 )
# lld = pickle.load( open('latlondict.p','r') )
# for row in feat_names[ feat_ranking == 3 ]:
#     n = int(row.split(' ')[-1])
#     lon,lat = lld[n]
#     plt.scatter( lon,lat, c='y', alpha=0.2 )
# for row in feat_names[ feat_ranking == 2 ]:
#     n = int(row.split(' ')[-1])
#     lon,lat = lld[n]
#     plt.scatter( lon,lat, c='g', alpha=0.2 )
# for row in feat_names[ feat_ranking == 1]:
#     n = int(row.split(' ')[-1])
#     lon,lat = lld[n]
#     plt.scatter( lon,lat, c='r', alpha=0.2 )
# plt.savefig('quickplot.png')
# 
# # now, you can explore which GEFs # goes with which point
# lld = pickle.load( open('latlongdict.p','r') )
# coords of feature i can be found with lld[i], and the coords of
# the zeroth MESONET point can be found with:
# mesonet_locs = np.recfromcsv('../../data/station_info.csv')[10]  # needed only for interpolated features
#
# RESULTS SUMMARY:
# Not entirely clear, but seems to point to the non-commented features (in the features class)
#  as being the most important ones.  Will keep those for now.
    
