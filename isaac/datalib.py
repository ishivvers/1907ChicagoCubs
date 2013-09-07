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

VARIABLE_NAMES = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm',
                  'tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']

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
    Should give a reasonable number to compare to the leaderboard scores.
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
            feats, name = f[1]()
            self.addfeat( feats, name )
        if scale:
            if self.verbose: print 'rescaling all input data'
            scl = StandardScaler()
            self.features = scl.fit_transform(self.features)
            self.scaler = scl
            
    def addfeat(self, features, name):
        '''
        Add an array of features with shape=(n_examples, n_GEFS_points, n_features) to the self.features
        array.
        '''
        if self.features == None:
            self.features = features
        else:
            self.features = np.dstack( (self.features,features) )
        self.featnames.append(name)
    
    def getshape(self):
        '''
        Returns the shape of the feature array.
        '''
        if self.features == None:
            return (0,)
        else:
            return self.features.shape
    
    def return_feats_near(self, n_mesonet, n=9, scale=True):
        '''
        Return the features for the <n> GEFs gridpoints nearest the <n_mesonet> station.
        '''
        lat,lon = self.mesonet_locs[n_mesonet][1], self.mesonet_locs[n_mesonet][2]
        sqdists = (self.GEFslat - lat)**2 + (self.GEFslon - lon)**2
        sort_sqdists = zip( sqdists, np.arange(len(sqdists)) )
        sort_sqdists.sort()
        indices_wanted = np.array([r[1] for r in sort_sqdists[:n]])
        
        features = self.features[:,indices_wanted,:]
        features = features.reshape( (features.shape[0], features.shape[1]*features.shape[2]) )
        if scale:
            if self.verbose: print 'rescaling all input data'
            scl = StandardScaler()
            features = scl.fit_transform(features)
            self.scaler = scl
        return features
    
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
        