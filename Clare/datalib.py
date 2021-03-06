'''
A library of python functions to help with 
the 2013 AMS Solar Energy Kaggle project.
'''

from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import numpy as np
from scipy.interpolate import interp2d
import csv
import pickle
import inspect
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
## imports added by Clare ##
import sunrise
import datetime
import scipy.interpolate
from copy import copy


VARIABLE_NAMES = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm',
                  'tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

# for example, plot up the cloud cover data
#f = Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
# radiation flux measure
#f = Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r')
#map_variable_movie(f)
def map_variable_movie(f, ens=0):
    '''
    Map variable in a GEFS file as a function of time.
    f: a netCDF4 Dataset object created from a GEFS file.
    ens: the index of the ensemble member to use
    '''
    var = f.variables.keys()[-1]    # the name of the variable encoded in f
    arr = f.variables[var]          # the variable array, accessed as var[...]
    days = f.variables['time'][:]
    hours = f.variables['fhour'][:]
    maxval = np.max(arr[:,ens,:,:,:]) # used to keep the plot size stable
    minval = np.min(arr[:,ens,:,:,:])

    plt.ion()                       # allows animation
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid( f.variables['lon'][:]-360, f.variables['lat'][:] )
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_zlabel(var)
    for i in range(arr.shape[0]):
        print i
        for j in range(arr.shape[2]):
            t = days[i] + hours[j] # the displayed time is hours since 1800-01-01 00:00:00
            Z = arr[i,ens,j,:,:]
            lines = ax.plot_wireframe(X, Y, Z)
            ax.set_zlim(minval, maxval)
            label = ax.annotate('time: '+str(t), (.7,.8), xycoords='figure fraction')
            plt.draw()
            #sleep(.05)            # uncomment to slow down animation, or if it freezes
            lines.remove()
            label.remove()



# for example, find the interpolated value at 265.5, 31.5 at 12:00 on the 10th day for the 1st ensemble member
#F = create_2d_interpolation(f, 10, 0, 1)
#value = F(265.5, 31.5)
def create_2d_interpolation(f, day, hour, ens=0):
    '''
    Returns a 2d spline function F(x,y) for the variable in the GEFS file, where
    x = lon and y = lat.
    f: a netCDF4 Dataset object created from a GEFS file.
    day: index of the day to use (labeled "time")
    hour: index of the hour to use (labeled "fhour")
    ens: index of the ensemble member to use (labeled "ens")
    '''
    var = f.variables.keys()[-1]             # the name of the variable encoded in f
    arr = f.variables[var][day,ens,hour,:,:] # the variable array, accessed as var[...]
    x = f.variables['lon'][:]-360
    y = f.variables['lat'][:]
    F = interp2d(x,y,arr, kind='linear', bounds_error=True)
    return F


# for example, return the modeled value nearest the point 265.5, 31.5 at 12:00 on the 10th day for the 1st ensemble member
#value = return_nearest_value(f, 265.5, 31.5, 10, 0, 1)
def return_nearest_value(f, lon, lat, day, hour, ens=0):
    '''
    Returns the nearest value to the position (lon,lat) for the variable in the GEFS file.
    f: a netCDF4 Dataset object created from a GEFS file.
    lon, lat: a position within the range of f
    day: index of the day to use (labeled "time")
    hour: index of the hour to use (labeled "fhour")
    ens: index of the ensemble member to use (labeled "ens")
    '''
    try:
        assert( (lon < f.variables['lon'][-1]-360) & (lon > f.variables['lon'][0]-360) ) # don't allow any requests outside of the grid
        assert( (lat < f.variables['lat'][-1]) & (lat > f.variables['lat'][0]) )
    except:
        raise Exception('Requested coordinates outside of file grid.')
    var = f.variables.keys()[-1]             # the name of the variable encoded in f
    x = np.round(lon)                        # just round to find the closest lat/long point
    y = np.round(lat)
    i = np.argmin( np.abs(f.variables['lon'][:]-360 - x) )
    j = np.argmin( np.abs(f.variables['lat'][:] - y) )
    val = f.variables[var][day,ens,hour,j,i]
    return val


def save_submission(predictions, name, data_dir='../../data/'):
    '''
    Save submission in the same format as the sampleSubmission.
    predictions: a numpy array of predictions.
    name: output file name.
    '''
    fexample = open( data_dir+'sampleSubmission.csv', 'r' )
    fout = open( data_dir+name, 'w' )
    fReader = csv.reader( fexample, delimiter=',', skipinitialspace=True )
    fwriter = csv.writer( fout )
    for i,row in enumerate( fReader ):
    	if i == 0:
    		fwriter.writerow( row )
    	else:
    		row[1:] = predictions[i-1]
    		fwriter.writerow( row )
    fexample.close()
    fout.close()


def load_MESONET(name, data_dir='../../data/'):
    '''
    Load a csv file as formatted for the MESONET files.
    Returns: times, Y
    '''
    data = np.loadtxt(data_dir+name, delimiter=',', dtype=float, skiprows=1)
    times = data[:,0].astype(int)
    Y = data[:,1:]
    return times,Y


def load_GEFS(which, variables, data_dir='../../data/', average_hours=True, average_models=True):
    '''
    Load up the GEFS files.
    which: can be one of "test" or "train"
    variables: a list of variable names to load, i.e. "apcp_sfc"
    average_*: take the mean value of that dimension in the GEFS file
    '''
    for each in variables:
        try:
            assert( each in VARIABLE_NAMES )
        except:
            raise Exception('bad variable name: '+each)
    if which == 'test':
        postfix = '_latlon_subset_20080101_20121130.nc'
    elif which == 'train':
        postfix = '_latlon_subset_19940101_20071231.nc'
    else:
        raise Exception('first argument must be either "test" or "train"')
    
    # for each variable, extract the array from the netCDF4 file
    all_variables = []
    for var in variables:
        X = Dataset( data_dir+which+'/'+var+postfix).variables.values()[-1][:]
        if average_hours and average_models:
            # average over models and hours, axes 1 and 2
            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4])
            X = np.mean(X, axis=1)
        elif average_models:
            # average over models only, axes 1
            X = np.mean(X, axis=1)
        elif average_hours:
            # average over hours only, axes 2
            X = np.mean(X, axis=2)
        # Reshape into standard sklearn format; shape = (n_timesteps, n_features)
        X = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        all_variables.append(X)
    
    # concatenate the variables array into the sklearn format
    #  should have shape = ( n_timesteps, n_variables*n_locations[*n_hours*n_models] )
    X = np.hstack( tuple(all_variables) )
    
    return X


def split_train( trainX, trainY, l=1796 ):
    '''
    Provides a quick way to seperate our training data into two sets
    to do internal comparisions.  Default is to slice off the length
    of the true test set (1796 values).
    Returns s_trainX, s_trainY, s_testX, s_testY
    '''
    return trainX[:-l], trainY[:-l], trainX[-l:], trainY[-l:]

def MAE( trueY, modelY ):
    '''
    Returns the Mean Absolute Error between arrays modelY and trueY
    (both should be 1D arrays).
    Use to test the quality of different models.
    
    For example, split your training set into two parts, reserving
    some fraction for a fake testing set.  Train your data only on
    the first part, and then produce modeled estimates for the second
    part.  Then use this function to compare between your model predictions
    and the true values for the fake testing set.
    
    Should give a reasonable number to compare to the leaderboard scores if 
    the fake testing set is the same length as the true.
    '''
    return metrics.mean_absolute_error(trueY, modelY)


class features:
    '''
    A class to handle all feature generation crap for the AMS Solar Energy Project.
    '''
    def __init__( self, which='train', verbose=False ):
        self.features = None
        self.scaler = None
        self.featnames = []
        self.which = which
        if self.which not in ['test','train']:
            raise Exception('which must be either test or train')
        self.verbose = verbose
        self.mesonet_locs = np.recfromcsv('../../data/station_info.csv')  # needed only for interpolated features
        
        # get arrays of lat & lon which correspond to the GEFs point locations
        if self.which == 'train':
            f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r')
        X,Y = np.meshgrid(f.variables['lon'][:], f.variables['lat'][:])
        self.GEFslat = Y.reshape( 9*16 )
        self.GEFslon = X.reshape( 9*16 ) - 360
        self.n_samples = f.variables['time'].shape[0]
        f.close()
    
    ##################################################################
    ## HELPER FUNCTIONS
    ##################################################################
    def save(self, fname):
        '''
        Save a pickled representation of this feature set.
        '''
        if self.verbose: print 'saving to file:', fname
        pickle.dump( self, open(fname,'w') )
    
    def load(self, fname):
        '''
        Load the feature set from a pickle file.
        '''
        if self.verbose: print 'importing from file:', fname
        other = pickle.load( open(fname,'r') )
        self.features, self.scaler, self.featnames, self.which, self.verbose, self.mesonet_locs = \
            other.features, other.scaler, other.featnames, other.which, other.verbose, other.mesonet_locs
    
    def integ(self, f):
        '''
        Integrates daily values over the variable in f, and averages over models.
        Returns a feature array.
        '''
        var = f.variables.keys()[-1]
        arr = np.mean(f.variables[var][:], axis=1) # average over all models
        X = f.variables['fhour'][:] * 3600
        i_arr = np.trapz( arr, X, axis=1 )         # integrate over hours
        features = i_arr.reshape( i_arr.shape[0], i_arr.shape[1]*i_arr.shape[2] )
        return features                            # return the array with shape=(n_timesteps, n_features)
    
    def calc_all_features(self, scale=True):
        '''
        Use the inspect module to calculate all features. Automatically
        runs all features defined with an underscore at the front of the name.
        '''
        feat_funcs = [f for f in inspect.getmembers(self) if  (f[0][0]=='_' and f[0][1]!='_' and inspect.ismethod(f[1]))]
        for f in feat_funcs:
            if self.verbose: print 'calculating',f[0]
            self.addfeat( f[1]()[0],f[1]()[1] )
        if scale:
            if self.verbose: print 'rescaling all input data'
            scl = StandardScaler()
            self.features = scl.fit_transform(self.features)
            self.scaler = scl
            
    def addfeat(self, features, name):
        '''
        Add an array of features with shape=(n_examples, n_features) to the self.features
        array.
        Uses the self.gefsmask array (must be a numpy boolean array) to use only features
        from certain GEFS stations.
        '''
        if self.features == None:
            self.features = features
        else:
            self.features = np.hstack( (self.features,features) )
        for i in range(features.shape[1]):
            self.featnames.append(name+' '+str(i))
    
    def getshape(self):
        '''
        Returns the shape of the feature array.
        '''
        if self.features == None:
            return (0,)
        else:
            return self.features.shape
    
    def calc_features_near(self, n_mesonet, n=8, scale=True):
        '''
        Calculate the features for the <n> GEFs gridpoints nearest the <n_mesonet> station.
        Automatically runs all features defined with an underscore at the front of the name.
        '''
        lat,lon = self.mesonet_locs[n_mesonet][1], self.mesonet_locs[n_mesonet][2]
        sqdists = (self.GEFslat - lat)**2 + (self.GEFslon - lon)**2
        sort_sqdists = zip( sqdists, np.arange(len(sqdists)) )
        sort_sqdists.sort()
        indices_wanted = np.array([r[1] for r in sort_sqdists[:n]])
        
        feat_funcs = [f for f in inspect.getmembers(self) if  (f[0][0]=='_' and f[0][1]!='_' and inspect.ismethod(f[1]))]
        for f in feat_funcs:
            if self.verbose: print 'calculating',f[0]
            feats, name = f[1]()
            self.addfeat( feats[:,indices_wanted], name )
        if scale:
            if self.verbose: print 'rescaling all input data'
            scl = StandardScaler()
            self.features = scl.fit_transform(self.features)
            self.scaler = scl
    
    def calc_interpolated_feats(self, n_mesonet, scale=True):
        '''
        Calculate the GEFs features interpolated to the <n_mesonet> station.
        Automatically runs all features defined with an underscore at the front of the name.
        '''
        lat,lon = self.mesonet_locs[n_mesonet][1], self.mesonet_locs[n_mesonet][2]
        
        feat_funcs = [f for f in inspect.getmembers(self) if  (f[0][0]=='_' and f[0][1]!='_' and inspect.ismethod(f[1]))]
        self.features = np.empty( (self.n_samples, len(feat_funcs)) )
        for n_feat,f in enumerate(feat_funcs):
            if self.verbose: print 'calculating and interpolating',f[0]
            feats, name = f[1]()
            for day in range(feats.shape[0]):
                F = interp2d(self.GEFslat, self.GEFslon, feats[day], kind='linear', bounds_error=True)
                self.features[day, n_feat] = F( lat, lon )
        if scale:
            if self.verbose: print 'rescaling all input data'
            scl = StandardScaler()
            self.features = scl.fit_transform(self.features)
            self.scaler = scl
    
    ##################################################################
    ## FEATURES
    ##################################################################
    def _IDSWF(self):
        '''
        Integrated Short-Wave Flux
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        f.close()
        return features, 'Int. SW Flux'

        
    def _IDSWFfY(self):
        '''
        Integrated Short-Wave Flux from Yesterday
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        newfeatures = np.empty_like(features)
        newfeatures[1:] = features[:-1]
        newfeatures[0] = features[0]
        f.close()
        return newfeatures, 'Int. SW Flux from Yesterday'
    
    def _IDSWFfT(self):
        '''
        Integrated Short-Wave Flux from Tomorrow
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        newfeatures = np.empty_like(features)
        newfeatures[:-1] = features[1:]
        newfeatures[-1] = features[-1]
        f.close()
        return newfeatures, 'Int. SW Flux from Tomorrow'
    
    def _IDLWF(self):
        '''
        Integrated Long-Wave Flux
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dlwrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dlwrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        f.close()
        return features, 'Int. LW Flux'
        
    def _IDLWFfY(self):
        '''
        Integrated Long-Wave Flux from Yesterday
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dlwrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dlwrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        newfeatures = np.empty_like(features)
        newfeatures[1:] = features[:-1]
        newfeatures[0] = features[0]
        f.close()
        return newfeatures, 'Int. LW Flux from Yesterday'
        
    def _IDLWFfT(self):
        '''
        Integrated Long-Wave Flux from Tomorrow
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/dlwrf_sfc_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/dlwrf_sfc_latlon_subset_20080101_20121130.nc','r')
        features = self.integ(f)
        newfeatures = np.empty_like(features)
        newfeatures[:-1] = features[1:]
        newfeatures[-1] = features[-1]
        f.close()
        return newfeatures, 'Int. LW Flux from Tomorrow'
    
    # def _MCC(self):
    #     '''
    #     Mean Cloud Cover
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.mean(arr, axis=1)                  # average over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     f.close()
    #     return features, 'Mean Cloud Cover'
    #     
    # def _MCCfY(self):
    #     '''
    #     Mean Cloud Cover from Yesterday
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.mean(arr, axis=1)                  # average over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[1:] = features[:-1]
    #     newfeatures[0] = features[0]
    #     f.close()
    #     return newfeatures, 'Mean Cloud Cover from Yesterday'
    # 
    # def _MCCfT(self):
    #     '''
    #     Mean Cloud Cover from Tomorrow
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.mean(arr, axis=1)                  # average over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[:-1] = features[1:]
    #     newfeatures[-1] = features[-1]
    #     f.close()
    #     return newfeatures, 'Mean Cloud Cover from Tomorrow'
    # 
    # def _AP(self):
    #     '''
    #     Accumulated Precipitation
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.sum(arr, axis=1)                   # sum over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     f.close()
    #     return features, 'Acc. Rain'
    # 
    # def _APfY(self):
    #     '''
    #     Accumulated Precipitation from Yesterday
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.sum(arr, axis=1)                   # sum over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[1:] = features[:-1]
    #     newfeatures[0] = features[0]
    #     f.close()
    #     return newfeatures, 'Acc. Rain from Yesterday'
    # 
    # def _APfT(self):
    #     '''
    #     Accumulated Precipitation from Tomorrow
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/tcdc_eatm_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/tcdc_eatm_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.sum(arr, axis=1)                   # sum over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[:-1] = features[1:]
    #     newfeatures[-1] = features[-1]
    #     f.close()
    #     return newfeatures, 'Acc. Rain from Tomorrow'    
    
    def _MT(self):
        '''
        Max Temperature
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/tmp_2m_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/tmp_2m_latlon_subset_20080101_20121130.nc','r')
        var = f.variables.keys()[-1]
        arr = np.max(f.variables[var][:], axis=2) # take the max value on the hours axis
        arr = np.mean(arr, axis=1)                # average over all models
        features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
        f.close()
        return features, 'Max Temp.'
    
    def _MTfY(self):
        '''
        Max Temperature from Yesterday
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/tmp_2m_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/tmp_2m_latlon_subset_20080101_20121130.nc','r')
        var = f.variables.keys()[-1]
        arr = np.max(f.variables[var][:], axis=2) # take the max value on the hours axis
        arr = np.mean(arr, axis=1)                # average over all models
        features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
        newfeatures = np.empty_like(features)
        newfeatures[1:] = features[:-1]
        newfeatures[0] = features[0]
        f.close()
        return newfeatures, 'Max Temp. from Yesterday'
    
    def _MTfT(self):
        '''
        Max Temperature from Tomorrow
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/tmp_2m_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/tmp_2m_latlon_subset_20080101_20121130.nc','r')
        var = f.variables.keys()[-1]
        arr = np.max(f.variables[var][:], axis=2) # take the max value on the hours axis
        arr = np.mean(arr, axis=1)                # average over all models
        features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
        newfeatures = np.empty_like(features)
        newfeatures[:-1] = features[1:]
        newfeatures[-1] = features[-1]
        f.close()
        return newfeatures, 'Max Temp. from Tomorrow'
        
    def _AP(self):
        '''
        Air Pressure
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/pres_msl_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/pres_msl_latlon_subset_20080101_20121130.nc','r')
        var = f.variables.keys()[-1]
        arr = np.mean(f.variables[var][:], axis=1)  # average over all models
        arr = np.mean(arr, axis=1)                  # average over all hours
        features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
        f.close()
        return features, 'Daily Mean Air Pressure'

    # def _APfY(self):
    #     '''
    #     Air Pressure from Yesterday
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/pres_msl_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/pres_msl_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.mean(arr, axis=1)                  # average over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[1:] = features[:-1]
    #     newfeatures[0] = features[0]
    #     f.close()
    #     return newfeatures, 'Daily Mean Air Pressure from Yesterday'
    # 
    # def _APfT(self):
    #     '''
    #     Air Pressure from Tomorrow
    #     '''
    #     if self.which == 'train':
    #         f=Dataset('../../data/train/pres_msl_latlon_subset_19940101_20071231.nc','r')
    #     else:
    #         f=Dataset('../../data/test/pres_msl_latlon_subset_20080101_20121130.nc','r')
    #     var = f.variables.keys()[-1]
    #     arr = np.mean(f.variables[var][:], axis=1)  # average over all models
    #     arr = np.mean(arr, axis=1)                  # average over all hours
    #     features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
    #     newfeatures = np.empty_like(features)
    #     newfeatures[:-1] = features[1:]
    #     newfeatures[-1] = features[-1]
    #     f.close()
    #     return newfeatures, 'Daily Mean Air Pressure from Tomorrow'

    def _dAP(self):
        '''
        Max 3hr Change in Air Pressure
        '''
        if self.which == 'train':
            f=Dataset('../../data/train/pres_msl_latlon_subset_19940101_20071231.nc','r')
        else:
            f=Dataset('../../data/test/pres_msl_latlon_subset_20080101_20121130.nc','r')
        var = f.variables.keys()[-1]
        arr = np.mean(f.variables[var][:], axis=1)  # average over all models
        arr = arr[:,1:,:,:] - arr[:,:-1,:,:]        # find the differences over the hours axis
        arr = np.max( arr, axis=1 )                 # find the max over the hours axis
        features = arr.reshape( arr.shape[0], arr.shape[1]*arr.shape[2] )
        f.close()
        return features, 'Max Change in Air Pressure'
        
    def sunrise_sunset(self,lat, lon, delta_hours):
        s = sunrise.sun( lat, 360- lon)
        if self.which == 'train':
            date_start = datetime.datetime(1994,1,1)
        elif self.which == 'test':
            date_start = datetime.datetime(2007,1,1, 2)
        else: date_start = datetime.datetime(2009,1,1,2)

        utc_time = date_start + datetime.timedelta( delta_hours/24)
        srise, sset = s.sunrise(when = utc_time), s.sunset(when = utc_time)
        return srise, sset

    def solar_fit_integ(self, hrs, values, stds):
        ww = np.where( stds == 0)
        weights = np.empty_like( stds )
        weights[1:-1] = 1/stds[1:-1]
        weights[ww] = max(weights) *3
        new_hours = np.linspace(hrs[0], hrs[-1], 20)
        
        spl = scipy.interpolate.splrep( hrs, values, w = weights,s=100, quiet = 1)
        #spleval = scipy.interpolate.splev( new_hours, spl)
        splinteg = scipy.interpolate.splint(hrs[0], hrs[-1], (spl[0],spl[1], 3) )
        return splinteg


    def _FITSWF(self):
        
        f = Dataset('../../data/%s/dswrf_sfc_latlon_subset_19940101_20071231.nc' % self.which,'r')

        var = f.variables.keys()[-1]
        arr = f.variables[var][:]
        lat, lon, dh, hours = f.variables['lat'][:], f.variables['lon'][:], f.variables['time'][:], f.variables['fhour'][:]
        dh -= dh[0]
        solar_flux_point = np.zeros((len(dh),len(lat), len(lon)))
        for i, h in enumerate(dh):
            for j, Lat in enumerate(lat):
                for k, Lon in enumerate(lon):
                    fit_hours = np.zeros(len(hours)+2)
                    fit_vals = np.zeros(len(hours)+2)
                    fit_stds = np.zeros(len(hours)+2)
                    srise, sset = self.sunrise_sunset( Lat, Lon, h)
                    if srise > 12: srise = 12
                    if sset < 24: sset = 24
                    fit_hours[0], fit_hours[-1] = srise - .5, sset+.5
                    fit_hours[1:-1] = hours
                    fit_vals[1:-1] = arr[i,0,:,j,k]
                    fit_stds[1:-1] = np.std(arr[i,:,:,j,k], axis = 0)
                    solar_fit = self.solar_fit_integ(fit_hours, fit_vals, fit_stds)
                    
                    if (( float('nan') == solar_fit) or (float('inf')== solar_fit)):
                        print fit_hours, fit_vals, fit_stds
                    solar_flux_point[i,j, k]= solar_fit
        features = solar_flux_point.reshape( len(dh),len(lat)*len(lon) )
        
        newfeatures = np.empty_like(features)
        newfeatures[:-1] = features[1:]
        newfeatures[-1] = features[-1]
        f.close()
        return newfeatures, 'Total solar flux fitting for sunrise and sunset'

