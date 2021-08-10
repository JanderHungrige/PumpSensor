#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:35:39 2021

@author: Jan Werth

Here we us the maniulated data which we analyzed in Sensor_analysis.py 


1# Here we first load the data.
2# Then we map the target strings to int
3# Scale the Sensor data between 0-1 with MinMax scaler
4# transform the timeseries into a supervised learning dataset. This is done by creating a moving window with the pandas shift function:
    Thanks to Jason Brownlee
5# Split the data in TRain Val Test based on visualinspection so that each has at least one failure in the set
6# One hot encode the target for class prediction. Signal prediction does not need the one hot encoding
7# Setup the model (use the functional api one). The model has two outputs. One for signal prediction and one for class prediction
8# Train or load the model
9# plot the results

toDo :    
  Attention: test_x  is also shited. meaning the plot does not account for the prediction?? 
  Result: test_y_raw did not show any difference!? interesting.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import sqrt
from Sensor_analysis import read_data, manipulate_X, Vorverarbeitung_Y
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from plotting_functions import plot_class_hat, plot_signal_hat, plot_training

class prepare_data:
    def __init__(self, data_x, data_y):
        self.data_x=data_x
        self.data_y=data_y
        self.encoded_y=[]
        self.oneHot=[]
        
    def Vorverarbeitung_Y(self):
        from sklearn import preprocessing
        #Label Mapping
        le = preprocessing.LabelEncoder()
        le.fit(self.data_y)
        self.data_y=le.transform(self.data_y)
        self.data_y=pd.DataFrame(self.data_y,columns=['target'])                
        #Get the Label map
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)

    def splitting(self, von, bis):    
        split_x=self.data_x[von:bis].values
        split_y=self.data_y[von:bis].values
        return split_x, split_y

    def make_float(self):      
        self.data_x.astype('float32')

    def reshape_for_Lstm(self):    
        # reshape for input 
        timesteps=1
        samples=int(np.floor(self.data_x.shape[0]/timesteps))   
        self.data_x=self.data_x.reshape((samples,timesteps,self.data_x.shape[1]))   #samples, timesteps, sensors     

    def one_hot(self):        #one hot encode the targets for class prediction/ not for signal prediction
        from sklearn.preprocessing import OneHotEncoder
        oneHot=OneHotEncoder()
        oneHot.fit(self.data_y.reshape(-1,1))
        self.oneHot=oneHot.transform(self.data_y.reshape(-1,1)).toarray()
       # return self.data_y
    
    def scaling(self,save=False):
        scaler=MinMaxScaler().fit(self.data_x)
        self.data_x=scaler.transform(self.data_x) 
        if save:
            from numpy import savetxt # save the data for later quantization
            savetxt('test_x_data_OOP.csv',self.data_x,delimiter='.')

class Zeitreihe:
    def __init__(self,data, n_in=1, n_out=1):
        self.data=data
        self.n_in=n_in
        self.n_out=n_out
    
    def series_to_supervised(self,dropnan=True):
        n_vars = 1 if type(self.data) is list else self.data.shape[1]
        df = pd.DataFrame(self.data)
        cols, namen = list(),list()
        # input sequence (t-n, ... t-1)
        for i in range(self.n_in, 0, -1):
            cols.append(df.shift(i))
            namen +=[('sensor%d(t-%d)' %(j+1, i)) for j in range (n_vars)]
            #forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.n_out):
            cols.append(df.shift(-i))
            if i == 0:
                namen +=[('sensor%d(t)' %(j+1)) for j in range (n_vars)]
            else:
                namen +=[('sensor%d(t+%d)' '%'(j+1, i)) for j in range (n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns=namen
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
            
        self.data=self.clean_series_to_supervised(agg)
        
        return self.data
    
    def clean_series_to_supervised(self,shifted_data):
        to_remove_list =['sensor'+str(n)+'(t)' for n in range(1,len(self.data.columns)+1)] #now remove all non shifted elements again. so we retreive elements and shifted target
        #to_remove_list_2 =['sensor'+str(n)+'(t-'+ str(i)+')' for n in range(1,len(data_scaled.columns)+1) for i in range(1,Future)] #now remove all non shifted elements again. so we retreive elements and shifted target
        #to_remove_list=to_remove_list_1+to_remove_list_2
        data_y=shifted_data.iloc[:,-1] #Get the target data out before removing unwanted data
        data_x=shifted_data.drop(to_remove_list, axis=1) #remove sensors(t)
        data_x.drop(data_x.columns[len(data_x.columns)-1], axis=1, inplace=True)# remove target(t-n)
        data=pd.concat([data_x,data_y],axis=1)
        data.columns=[*data.columns[:-1],'machine_status'] # rename last column to target name
        return data

# %%
#MODEL DEFINITIONS
    
def model_setup_Fapi(in_shape):
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    
    inputs= tf.keras.Input(shape=(in_shape[1],in_shape[2]))
    x=LSTM(42,activation='relu', input_shape=(in_shape[1],in_shape[2]),return_sequences=True)(inputs)
    x=LSTM(42,activation='relu')(x)
    out_signal=Dense(1, name='signal_out')(x)
    out_class=Dense(3,activation='softmax', name='class_out')(x)
    
    model=tf.keras.Model(inputs=inputs, outputs=[out_signal,out_class])
    
    model.compile(loss={'signal_out':'mean_squared_error',
                        'class_out' :'categorical_crossentropy'},
                         optimizer='adam',
                         metrics={'class_out':'acc'})
    
    print(model.summary())
    return model
   
#%%    
if __name__ == '__main__':
    
#LOAD DATA    
    data,sensorname=read_data('pump_sensor.csv')
    
    #PREPROCESS DATA   

    data=manipulate_X(data, printplot=False); sensorname=data.keys()[2:-1] 
    data=data[sensorname.insert(len(sensorname),'machine_status')] # only sensors and target
    
#CREATE WINDOWED DATA    
    FUTURE=1
    Timeseries=Zeitreihe(data,FUTURE).series_to_supervised()
    sensornames_shift=Timeseries.keys()[:-1]
    
#PREPROCESS DATA
    Data=prepare_data(Timeseries[sensornames_shift], Timeseries['machine_status']) #Create class instance of Data
    Data.Vorverarbeitung_Y() # change Y to mapped target values
    Data.make_float()
    
    #splitting data into trainval test
    train_x,train_y=Data.splitting(0,120000)
    val_x,val_y    =Data.splitting(140000,len(Data.data_x))
    test_x,test_y  =Data.splitting(120000,140000)
    
    Train_data=prepare_data(train_x, train_y) #create class instances
    Val_data  =prepare_data(val_x, val_y)
    Test_data =prepare_data(test_x, test_y)
    
    Train_data.one_hot()
    Train_data.scaling()
    Train_data.reshape_for_Lstm()
    
    Val_data.one_hot()
    Val_data.scaling()
    Val_data.reshape_for_Lstm()
    
    Test_data.one_hot()
    Test_data.scaling(save=True)
    Test_data.reshape_for_Lstm()
    

# %%
# TRAIN THE MODEL...    
    TRAIN=True
    EPOCH=20
    BATCHSIZE=32
    
    inputshape_X=(Train_data.data_x.shape)
    #print(inputshape_X)
    
    if TRAIN:
        #model=model_setup_seq(inputshape_X)
        #history = model.fit(train_X, train_Y, epochs=80, batch_size=32, validation_data=(val_X, val_Y), shuffle=False)
    
        model=model_setup_Fapi(inputshape_X)
        history = model.fit(Train_data.data_x, [Train_data.data_y, Train_data.oneHot], epochs=EPOCH, batch_size=BATCHSIZE, validation_data=(Val_data.data_x, [Val_data.data_y,Val_data.oneHot]), shuffle=False)
        plot_training([history.history['class_out_loss'],history.history['val_class_out_loss']],
                      what='loss',
                      saving=True,
                      name=('training_'+ str(FUTURE)))  
        plot_training([history.history['class_out_acc'],history.history['val_class_out_acc']],
                      what='acc',
                      saving=True,
                      name=('training_'+ str(FUTURE))) 
        model.save('./model/Pump_LSTM_Fapi_OOP_'+ str(FUTURE))
        
# ...OR LOAD THE MODEL  
    else:  
        model=tf.keras.models.load_model('./model/Pump_LSTM_Fapi')
        
# %%    
# INFERENCE
    # make a prediction
    [yhat,yclass] = model.predict(Test_data.data_x)    
    Yclass=[np.argmax(yclass[i],0) for i in range(len(yclass))] # get final class
       
    plot_signal_hat(yhat,Test_data.data_y,saving=True, name='Prediction_Signal_fapi3_42_'+ str(FUTURE))
    plot_signal_hat(Yclass,Test_data.data_y,saving=True, name='Prediction_class_fapi3_42_'+ str(FUTURE))
    