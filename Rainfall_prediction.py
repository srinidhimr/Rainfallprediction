#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from netCDF4 import Dataset#pip install netCDF4


# In[9]:


path='E:/Rain_Metrics/data_gpcp/'


# In[14]:


years=np.arange(1979,2016)
months=np.arange(1,13)
print(months)
print(years)


# In[18]:


rain=np.array((0));
ym=np.array((0));
for year in years:
    for month in months:
        if len(str(month))==1:
            month_str='0'+str(month)
        else:
            month_str=str(month)
        sub_path=path+ str(year)  + month_str + '.nc'
        #print(url)
        data=Dataset(sub_path)
        precip=data.variables['precip']
        var=precip[:,:,:]
        temp=var.reshape(var.shape[0],-1).T
        if rain.shape==():
            rain=temp;
            ym=int(str(year)+month_str)
        else:
            rain=np.hstack((rain,temp))
            ym=np.hstack((ym,int(str(year)+month_str)))
        #rain=pd.DataFrame(temp1.T)
    print('Shape='+str(rain.shape))


# In[20]:


ym.shape


# In[21]:


rain_data=pd.DataFrame(rain, columns=ym)
rain_data


# In[22]:


rain_data.describe()


# In[40]:


rain_data[201501].hist()


# In[48]:


rain_data[201506].hist();


# In[65]:


from matplotlib import pyplot
#rain_data[[201501,201502,201503,201504,201505,201506,201507,201508,201509,201510,201511,201512]].plot()
rain_data[[201501,201507]].plot()
pyplot.show()


# In[69]:


import seaborn as sns
pyplot.figure(figsize=(11,4))
sns.heatmap(rain_data[[201501,201502,201503,201504,201505,201506,201507,201508,201509,201510,201511,201512]].corr(),annot=True)
pyplot.show()


# In[74]:


division_data=np.asarray(rain_data)
division_data.shape


# In[76]:


# seperation of training and testing data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X = None; y = None
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[78]:


y_test.shape


# In[79]:


data_2010=np.asarray(rain_data[[201001,201002,201003,201004,201005,201006,201007,201008,201009,201010,201011,201012]])
X_year_2010 = None; y_year_2010 = None
for i in range(data_2010.shape[1]-3):
    if X_year_2010 is None:
        X_year_2010 = data_2010[:, i:i+3]
        y_year_2010 = data_2010[:, i+3]
    else:
        X_year_2010 = np.concatenate((X_year_2010, data_2010[:, i:i+3]), axis=0)
        y_year_2010 = np.concatenate((y_year_2010, data_2010[:, i+3]), axis=0)


# In[80]:


X_year_2010


# In[81]:


data_2015=np.asarray(rain_data[[201501,201502,201503,201504,201505,201506,201507,201508,201509,201510,201511,201512]])
X_year_2015 = None; y_year_2015 = None
for i in range(data_2015.shape[1]-3):
    if X_year_2015 is None:
        X_year_2015 = data_2015[:, i:i+3]
        y_year_2015 = data_2015[:, i+3]
    else:
        X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i+3]), axis=0)
        y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i+3]), axis=0)


# In[84]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
from tensorflow.keras import backend
# NN model
inputs = Input(shape=(3,1))
x = Conv1D(64, 2, padding='same', activation='relu')(inputs)
x = Conv1D(128, 2, padding='same', activation='relu')(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model = Model(inputs=[inputs], outputs=[x])
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
model.summary()


# In[86]:


model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
y_pred = model.predict(np.expand_dims(X_test, axis=2))
print (mean_absolute_error(y_test, y_pred))


# In[149]:


pyplot.plot(y_test,'.', y_pred,'-')
pyplot.xlabel('Locations')
pyplot.ylabel('Average Rainfall (mm)')
pyplot.legend(['Actual Values','Predicted Values'])


# In[102]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

inputs = Input(shape=(3,1))
x = LSTM(64, activation='relu')(inputs)
#x = LSTM(128, activation='relu')(x)
#x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model_lstm = Model(inputs=[inputs], outputs=[x])
model_lstm.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
model_lstm.summary()


# In[103]:


model_lstm.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.1, shuffle=True)
y_pred = model_lstm.predict(np.expand_dims(X_test, axis=2))
print (mean_absolute_error(y_test, y_pred))


# In[150]:


pyplot.plot(y_test,'.', y_pred,'-')
pyplot.xlabel('Locations')
pyplot.ylabel('Average Rainfall (mm)')
pyplot.legend(['Actual Values','Predicted Values'])


# In[155]:


data_2010.shape


# In[118]:




#2010
y_year_pred_2010 = model_lstm.predict(np.expand_dims(X_year_2010,axis=2))
    
#2015
y_year_pred_2015 = model_lstm.predict(np.expand_dims(X_year_2015,axis=2))



print ("MEAN 2010")
print (np.mean(y_year_2010),np.mean(y_year_pred_2010))
print ("Standard deviation 2010")
print (np.sqrt(np.var(y_year_2010)),np.sqrt(np.var(y_year_pred_2010)))


print ("MEAN 2015")
print (np.mean(y_year_2015),np.mean(y_year_pred_2015))
print ("Standard deviation 2015")
print (np.sqrt(np.var(y_year_2015)),np.sqrt(np.var(y_year_pred_2015)))



# In[148]:


pyplot.plot(y_year_2010,'.', y_year_pred_2010,'-')
pyplot.xlabel('Locations')
pyplot.ylabel('Average Rainfall (mm)')
pyplot.legend(['Actual Values','Predicted Values'])


# In[151]:


#Rainfall Prediction of Karnataka
data=Dataset('E:/Rain_Metrics/data_gpcp/201506.nc')
lat=data.variables['latitude'][:]
lon=data.variables['longitude'][:]
karnataka_lat=16.31
karnataka_lon=75.71
sq_diff_lat=(lat-karnataka_lat)**2
sq_diff_lon=(lon-karnataka_lon)**2
min_index_lat=sq_diff_lat.argmin()
min_index_lon=sq_diff_lon.argmin()
print(min_index_lat)
print(min_index_lon)
print(lat)


# In[153]:


raink=np.array((0));
flag=0
for year in years:
    for month in months:
        if len(str(month))==1:
            month_str='0'+str(month)
        else:
            month_str=str(month)
        sub_path=path+ str(year)  + month_str + '.nc'
        #print(url)
        data=Dataset(sub_path)
        precip=data.variables['precip']
        var=precip[0,min_index_lat,min_index_lon].data
        temp=var;
       # print(temp)
        #temp=var.reshape(var.shape[0],-1).T
        if flag==0:
            raink=temp;
            #ym=int(str(year)+month_str)
            flag=1
        else:
            raink=np.hstack((raink,temp))
            #ym=np.hstack((ym,int(str(year)+month_str)))
        #rain=pd.DataFrame(temp1.T)
print('Shape='+str(raink.shape))


# In[165]:


raink.shape


# In[167]:


from numpy import array
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):

		end_ix = i + n_steps

		if end_ix > len(sequence)-1:
			break

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

Xk, yk = split_sequence(list(raink), 3)


# In[173]:


y_pred_kar = model_lstm.predict(np.expand_dims(Xk,axis=2))
print ("MEAN Karnataka")
print (np.mean(yk),np.mean(y_pred_kar))
print ("Standard deviation Karnataka")
print (np.sqrt(np.var(yk)),np.sqrt(np.var(y_pred_kar)))


# In[176]:


pyplot.plot(yk,'*', y_pred_kar,'-')
pyplot.xlabel('Months')
pyplot.ylabel('Average Rainfall (mm)')
pyplot.legend(['Actual Values','Predicted Values'])


# In[ ]:




