import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import statistics
import sys
from tensorflow import keras
from keras.models import Sequential
from keras.layers import MaxPooling1D, SimpleRNN,Dropout
from keras.callbacks import EarlyStopping
from numpy import array
from sklearn.metrics import mean_squared_error


#Choose dataset Geant or Abilene
if len(sys.argv)==1:
   print('Choose dataset between GEANT or ABILENE')
   exit()
else:
   dataset = sys.argv[1]
  
# Load the data
if dataset=='GEANT':
   traffic_matrix=np.load('data/GEANT/X.npy')
elif dataset=='ABILENE':   
   traffic_matrix=np.load('data/ABILENE/X.npy')
else:
   print('WRONG dataset! Choose between GEANT or ABILENE')
   exit()
data = traffic_matrix[:, :, :].copy()


# Parameters of our time series
SPLIT_TIME = int(len(data) * 0.8)
WINDOW_IN = 10
WINDOW_OUT = 1
NODES=data.shape[2]


# Normalize the data
if dataset==('GEANT'):
   data = np.clip(data, 0.0, np.percentile(data.flatten(), 99.9))#geant
elif dataset==('ABILENE'):
   data = np.clip(data, 0.0, np.percentile(data.flatten(), 99.99998))#abilene
data_split = data[:SPLIT_TIME]
max_list = np.max(data)
min_list = np.min(data)
data = (data - min_list) / (max_list - min_list)
data[np.isnan(data)] = 0  # fill the abnormal data with 0
data[np.isinf(data)] = 0


# define input sequence
# split a multivariate sequence into x=[samples, window_in, rows, columns] y=[samples, window_out, rows, columns]
def split_sequences(sequences, WINDOW_IN, WINDOW_OUT):
   X, y = list(), list()
   for i in range(len(sequences)):
       # find the end of this pattern
       end_ix = i + WINDOW_IN
       out_end_ix = end_ix + WINDOW_OUT
       # check if we are beyond the dataset
       if out_end_ix > len(sequences):
           break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequences[i:end_ix, :, :], sequences[end_ix:out_end_ix, :, :]
       X.append(seq_x)
       y.append(seq_y)
   return array(X), array(y)


# convert into input/output all the samples
X, y = split_sequences(data, WINDOW_IN, WINDOW_OUT)


# reshape from [samples, window, rows, columns] into [samples, window, rows*columns]
X = X.reshape(X.shape[0], WINDOW_IN,NODES*NODES)
y = y.reshape(y.shape[0], WINDOW_OUT,NODES*NODES)


# Split into training data and test data
series_train_x = X[:SPLIT_TIME]
series_train_y = y[:SPLIT_TIME]
series_test_x = X[SPLIT_TIME:]
series_test_y = y[SPLIT_TIME:]


# Define model
model = Sequential()
model.add(SimpleRNN(512, return_sequences = True))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(SimpleRNN(256, return_sequences = True))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=5))
model.add(SimpleRNN(NODES*NODES, return_sequences = True))


model.compile(loss='mae', optimizer='adam', metrics=["accuracy"]) 


early_stopping = EarlyStopping(monitor='loss',patience = 10)


# fit model (data = batch_size*samples per epoch)
model.fit(series_train_x, series_train_y, epochs=100, batch_size=128, verbose=1, callbacks=[early_stopping])
model.summary()


# Prediction on the test series
val_forecast = model.predict(series_test_x, verbose=0)


# Rescale to original values
series_test_y=series_test_y*(max_list - min_list) + min_list
val_forecast=val_forecast*(max_list - min_list) + min_list


# calculate RMSE, NMAE, TRE, SRE
series_test_y=series_test_y.reshape(series_test_y.shape[0],series_test_y.shape[1]*series_test_y.shape[2])
val_forecast=val_forecast.reshape(val_forecast.shape[0],val_forecast.shape[1]*val_forecast.shape[2])
rmse=list()
nmae=list()
tre=list()
sre=list()


for t in range(series_test_y.shape[0]):
   rmse.append(np.sqrt(mean_squared_error(series_test_y[t,:], val_forecast[t,:]))/1000)
   nmae.append(sum(np.absolute(np.subtract(series_test_y[t,:],val_forecast[t,:])))/sum(np.absolute(series_test_y[t,:])))
   tre.append(np.sqrt(sum((np.subtract(series_test_y[t,:],val_forecast[t,:]))**2))/np.sqrt(sum(series_test_y[t,:]**2)))


for i in range(series_test_y.shape[1]):
   if(np.sqrt(sum(series_test_y[:,i]**2))!=0):
       result = np.sqrt(sum((np.subtract(series_test_y[:,i],val_forecast[:,i]))**2))/np.sqrt(sum(series_test_y[:,i]**2))
       sre.append(result)
   elif(np.sqrt(sum(series_test_y[:,i]**2))==0):
       result = 0
       sre.append(result)


# Printing the mean
print('Mean RMSE Test Score: %.4f Mbps' % (np.mean(rmse)))
print('Median RMSE Test Score: %.4f Mbps' % (statistics.median(rmse)))
print('Std RMSE Test Score: %.4f Mbps' % (np.std(rmse)))
print('Max RMSE Test Score: %.4f Mbps' % (max(rmse)))


print('Mean NMAE Test Score: %.4f' % (np.mean(nmae)))
print('Median NMAE Test Score: %.4f' % (statistics.median(nmae)))
print('Std NMAE Test Score: %.4f' % (np.std(nmae)))
print('Max NMAE Test Score: %.4f' % (max(nmae)))


print('Mean TRE Test Score: %.4f' % (np.mean(tre)))
print('Median TRE Test Score: %.4f' % (statistics.median(tre)))
print('Std TRE Test Score: %.4f' % (np.std(tre)))
print('Max TRE Test Score: %.4f' % (max(tre)))


print('Mean SRE Test Score: %.4f' % (np.mean(sre)))
print('Median SRE Test Score: %.4f' % (statistics.median(sre)))
print('Std SRE Test Score: %.4f' % (np.std(sre)))
print('Max SRE Test Score: %.4f' % (max(sre)))


# Plot 1
plt.figure(1,figsize=(10, 6))
if dataset==('ABILENE'):   
   plt.title("Traffic Prediction of Node2 'ATLAng' to Node9 'NYCMng'")#Abilene
   plt.plot(np.squeeze(series_test_y[:,20]), label="validation set")#Abilene
   plt.plot(np.squeeze(val_forecast[:,20]), label="predicted")#Abilene
elif dataset==('GEANT'):
   plt.title("Traffic Prediction of Node '2' to Node '8'")#GEANT
   plt.plot(np.squeeze(series_test_y[:,30]), label="validation set")#GEANT
   plt.plot(np.squeeze(val_forecast[:,30]), label="predicted")#GEANT
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.show()


# Plot 2
plt.figure(2,figsize=(10, 6))
plt.title("Root Mean Square Error through time")
plt.plot(rmse, 'r', label="RMSE")
plt.xlabel("Timestep")
plt.ylabel("Root Mean Square Error (Mbps)")
plt.legend()
plt.show()


# Plot 3
plt.figure(3,figsize=(10, 6))
plt.title("Normalized Mean Absolute Error through time")
plt.plot(nmae,'r', label="NMAE")
plt.xlabel("Timestep")
plt.ylabel("Normalized Mean Absolute Error")
plt.legend()
plt.show()


# Plot 4
plt.figure(4,figsize=(10, 6))
plt.title("Temporal Relative Error through time")
plt.plot(tre, 'r', label="TRE")
plt.xlabel("Timestep")
plt.ylabel("Temporal Relative Error")
plt.legend()
plt.show()


# Plot 5
plt.figure(5,figsize=(10, 6))
plt.title("Spatial Relative Error per OD flow")
plt.plot(sre, 'r', label="SRE")
plt.xlabel("OD Flow")
plt.ylabel("Spatial Relative Error")
plt.legend()
plt.show()