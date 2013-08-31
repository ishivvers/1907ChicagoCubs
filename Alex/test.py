import netCDF4
import numpy as np
import datalib
import matplotlib.pyplot as plt


def eval(actual, prediction):
    return np.mean(np.absolute(prediction-actual)) / 1000000

#def main():
datapath="../../data/"
#import the location information of the stations (name, lat, lon, elev)
stationData = np.genfromtxt(datapath+"station_info.csv",delimiter=',',dtype=[("stid","S4"),("nlat",float),("elon",float),("elev",float)],skiprows=1)

#import actual solar fluxes
y = np.genfromtxt(datapath+"train.csv",delimiter=',',skiprows=1,dtype=float)
times = y[:,0]  #gets the dates

#restrict our attention to one station
sIndex = 0; # station number
y = y[:,sIndex+1] #takes the y solar flux for the first stationsDat = stationData[sIndex] #gets the name,location of the first station
sDat=stationData[sIndex]
print sDat
    
# Prepare paths to different files
VARIABLE_NAMES = ['apcp_sfc','dlwrf_sfc','dswrf_sfc','pres_msl','pwat_eatm','spfh_2m','tcdc_eatm',
                      'tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc','ulwrf_sfc','ulwrf_tatm','uswrf_sfc']
n = len(VARIABLE_NAMES) # n is typically used for the number of features

#paths to the different files
DATA_NAMES=[]  
for var in VARIABLE_NAMES:
    DATA_NAMES.append(datapath+ 'train/'+var+'_latlon_subset_19940101_20071231.nc')
    
#load data
data=netCDF4.Dataset(DATA_NAMES[0])

m = len(data.variables['time'][:]) # m is typically used for the number of training examples

# Use less of the data for the sake of speed

n = n
#m = 5 * m / 14
m=m
y = y[0:m]

# Parse Data by interpolating the data for the site that we are interested in and time averaging the various quantities over the days
# prepare lat / lon interpolation
lons = data.variables['lon'][:] - 360  #longitude in data and in station info don't match
lats = data.variables['lat'][:]

from scipy.interpolate import interp2d
X = np.zeros((m,n))
for dataIndex in range(n):
    data = netCDF4.Dataset(DATA_NAMES[dataIndex])
    ens=1 #choose ensemble member zero
    Y = np.mean(data.variables.values()[-1][:,ens,:,:,:],axis=1)
    for timeIndex in range(m):
        F = interp2d(lons,lats,Y[timeIndex],kind='linear',bounds_error=True)
        X[timeIndex,dataIndex] = F(sDat[2],sDat[1]) # make sure lon / lat are correctly ordered

# Scale Features
from sklearn import preprocessing
X = preprocessing.scale(X)

print "Data has been loaded"

# Load Cross Validation tools
from sklearn.cross_validation import KFold
    
kf = KFold(m, n_folds=5, indices=False)


#print "Linear Regression"
# Load a Linear Regression Model
from sklearn import linear_model
#clf = linear_model.LinearRegression()
clf = linear_model.RidgeCV(alphas=[0.01, 0.025, 0.05])

train_errors = []
test_errors = []
alpha_choices = []
for train, test in kf:
    clf.fit(X[train],y[train])
    train_errors.append( eval( y[train], clf.predict( X[train] ) ) )
    test_errors.append( eval( y[test], clf.predict( X[test] ) ) )
    alpha_choices.append(clf.alpha_)



#plt.ion()
#plt.plot(pred)
#plt.plot(y)
#plt.show()

print train_errors
print test_errors
print alpha_choices

#if __name__ == "__main__":
#    main()