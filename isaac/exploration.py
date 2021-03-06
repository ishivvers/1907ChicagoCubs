'''
A set of scripts/functions that are helpful when looking at the data.
'''

from datalib import *

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
    fig = plt.figure( figsize=(10,8) )
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
            ax.set_zlim(minval, .5*maxval)
            label = ax.annotate('time: '+str(t), (.7,.8), xycoords='figure fraction')
            plt.draw()
            #sleep(.05)            # uncomment to slow down animation, or if it freezes
            lines.remove()
            label.remove()

def plot_interpolated(f=Dataset('../../data/test/dswrf_sfc_latlon_subset_20080101_20121130.nc','r'),
                        day=1000, hour=2, ens=0):
    '''
    Plot up the GEFS variable and the interpolation/nearest neighbor.
    '''
    mesonet = np.recfromcsv('../../data/station_info.csv')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # the GEFS grid
    var = f.variables.keys()[-1]        # the name of the variable encoded in f
    X,Y = np.meshgrid( f.variables['lon'][:]-360, f.variables['lat'][:] )
    Z = f.variables[var][day,ens,hour,:,:]  # 18:00 on the 1000th day for the 0th ensemble
    ax.plot_wireframe(X, Y, Z)
    # the MESONET points
    F = create_2d_interpolation(f, day, hour, ens)
    for stid, lat, lon, elev in mesonet:
        # interpolated
        z = F(lon,lat)
        ax.scatter( lon, lat, z, c='g', alpha=0.75 )
        # nearest
        z = return_nearest_value( f, lon, lat, 1000, 2, 0 )
        ax.scatter( lon, lat, z, c='r', alpha=0.75, marker='x' )
    
    ax.set_xlabel('lon')
    ax.set_ylabel('lat')
    ax.set_zlabel(var)
    ax.set_title('green: interp. red: nearest')
    plt.show()

def compare_to_model_flux(f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r'), minindex=None, maxindex=None):
    '''
    Integrate the daily GEFS shortwave flux (averaged over models and 
    interpolated to MESONET locations) and compare to the training data.
    If minindex or maxindex is given, only plots out that range (this is time-intensive!).
    '''
    var = f.variables.keys()[-1]
    mesonet_train = np.recfromcsv('../../data/train.csv')
    mesonet_locs = np.recfromcsv('../../data/station_info.csv')
    if minindex==None:
        minindex = 0
    if maxindex==None:
        maxindex = f.variables[var].shape[0]
    
    # make a dictionary of lists of daily errors for each station
    errors = {}
    
    print 'calculating errors:',maxindex-minindex,'total'
    for i,day in enumerate(xrange(minindex,maxindex)):
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
        # integrate the daily flux vector and append to the errors dictionary
        #  fluxes are in joules per second per meter squared, so we should
        #  integrate over time in seconds
        for j,stid in enumerate(daily_fluxes.keys()):
            x = f.variables['fhour'][:] * 3600
            y = daily_fluxes[stid]
            model_integrated_flux = np.trapz(y,x)
            true_integrated_flux = mesonet_train[i][j+1]
            if stid in errors.keys():
                errors[stid][0].append( true_integrated_flux )
                errors[stid][1].append( model_integrated_flux )
            else:
                errors[stid] = [ [true_integrated_flux], [model_integrated_flux] ]

    # finally, make a plot!
    print 'plotting it up!'
    fig,axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 6) )
    x = f.variables['time'][minindex:maxindex]
    for stid in errors.keys():
        y = np.array(errors[stid])
        axs[0].plot( x, y[0], alpha=0.5 )
        axs[1].plot( x, y[1], alpha=0.5 )
        axs[2].plot( x, y[0]-y[1], alpha=0.5 )
    axs[2].set_xlabel('Hours since 00:00:00 01/01/1800')
    axs[0].set_ylabel('True Daily Energy (j/m^2)')
    axs[1].set_ylabel('Modeled Integrated Flux (j/m^2)')
    axs[2].set_ylabel('Difference (j/m^2)')
    axs[0].set_title('Measured and Modeled Daily Energy')
    plt.show()


