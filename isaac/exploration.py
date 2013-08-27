'''
A set of scripts/functions that are helpful when looking at the data.
'''

from datalib import *


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

def integrate_daily_flux(f=Dataset('../../data/train/dswrf_sfc_latlon_subset_19940101_20071231.nc','r'), maxindex=None):
    '''
    Integrate the daily GEFS shortwave flux (averaged over models and 
    interpolated to MESONET locations) and compare to the training data.
    If maxindex is given, only plots out that many days (this is time-intensive!).
    '''
    var = f.variables.keys()[-1]
    mesonet_train = np.recfromcsv('../../data/train.csv')
    mesonet_locs = np.recfromcsv('../../data/station_info.csv')
    if maxindex==None:
        maxindex = f.variables[var].shape[0]
    
    # make a dictionary of lists of daily errors for each station
    differences = {}
    for station in mesonet_locs:
        differences[station[0]] = []
    
    print 'calculating differences:',maxindex,'total'
    for i,day in enumerate(xrange(maxindex)):
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
        # integrate the daily flux vector and append to the differences dictionary
        #  fluxes are in joules per second per meter squared, so we should
        #  integrate over time in seconds
        for j,stid in enumerate(daily_fluxes.keys()):
            x = f.variables['fhour'][:] * 3600
            y = daily_fluxes[stid]
            model_integrated_flux = np.trapz(y,x)
            true_integrated_flux = mesonet_train[i][j+1]
            if stid in differences.keys():
                differences[stid].append( true_integrated_flux-model_integrated_flux )
            else:
                differences[stid] = true_integrated_flux-model_integrated_flux

    # finally, make a plot!
    print 'plotting it up!'
    plt.figure( figsize=(12, 6) )
    x = f.variables['time'][:maxindex]
    for stid in differences.keys():
        y = differences[stid]
        plt.plot( x, y, alpha=0.5 )
    plt.xlabel('Hours since 00:00:00 01/01/1800')
    plt.ylabel('True - Integrated Model (j m^2)')
    plt.show()

