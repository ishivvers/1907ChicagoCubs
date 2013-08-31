"""
A script to set-up a learning algorithm;
Initial simplifications:
    - day = 19940101
    - ensemble = 0
    - hour = 12 (?)
"""
import datalib
import numpy as np
import netCDF4
from glob import glob
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
import sep_utility

day = 19940101
ens = 0
hour = 18

# Get all of the GEFS data and make into functions:
gefs_files = glob('../../data/train/*.nc')
gefs_data = [netCDF4.Dataset(f) for f in gefs_files]
gefs_functions = [datalib.create_2d_interpolation(f, day*100, hour, ens =0)
                  for f in gefs_data]


# Get the station data:
stationFilename = '../../data/station_info.csv'
stationData = np.genfromtxt(stationFilename,delimiter=',',
                            dtype=[("stid","S4"),("nlat",float),("elon",float),("elev",float)],skiprows=1)
lat_lon = np.array([stationData['nlat'], stationData['elon']]).T

# Interpolate features from GEFS data to station data:
station_features = np.zeros((len(lat_lon),len(gefs_files)))
for i, gefs_fn in enumerate(gefs_functions):
    station_features[:,i] = np.array([gefs_fn( ll[0],ll[1] ).T[0] for ll in lat_lon])

# Get the training data:
trainingData = np.loadtxt('../../data/train.csv', delimiter = ',', skiprows = 1)
testdate = np.where(trainingData[:,0] == day )
training_data = trainingData[testdate,1:][0][0]

# Now run Random Forest Regressor.
clf = RFR( n_estimators = 20, n_jobs = -1 )

test_clf = clf.fit( station_features, training_data.T )
model = test_clf.predict( station_features )
print test_clf.score( station_features, training_data.T )
print model, training_data.T
print sep_utility.evaluate( model, training_data.T )
    
"""
To do:
Done - Fix interpolation of features to stations
Done - Read in training data, ../../data/train.csv
Done - Run RFC.
Predict on test set.
"""
