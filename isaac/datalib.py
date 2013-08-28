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
    
    
    
    
    