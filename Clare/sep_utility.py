
import numpy as np

def evaluate( model, data ):
    """
    A function to get mean absolute error in order to evaluate the model
    s and e are the number of sites and events (days)
    """
    
    s,e = data.shape

    abs_dif = abs(data - model)
    
    return np.sum( abs_dif)/(s*e)

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
    dd = np.where( f.variables['intTime'][:] == day)[0]
    hh = np.where( f.variables['fhour'][:] == hour )[0]
    arr = f.variables[var][dd,ens,hh,:,:] # the variable array, accessed as var[...]
    x = f.variables['lon'][:]-360
    y = f.variables['lat'][:]
    F = interp2d(x,y,arr, kind='linear', bounds_error=True)
    return F
