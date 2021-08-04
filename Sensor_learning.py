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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from math import sqrt
from Sensor_analysis import read_data, manipulate_X, Vorverarbeitung_Y
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, namen = list(),list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        namen +=[('sensor%d(t-%d)' %(j+1, i)) for j in range (n_vars)]
        #forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
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
    return agg

def splitting_and_shape_data(data_x,data_y):    
    train_X=data_x[0:120000].values
    train_Y=data_y[0:120000].values
    
    val_X=data_x[140000::].values
    val_Y=data_y[140000::].values
    
    test_X=data_x[120000:140000].values
    test_Y=data_y[120000:140000].values
      
    train_X.astype('float32')
    val_X.astype('float32')
    test_X.astype('float32')
    
    return train_X,train_Y,val_X,val_Y,test_X,test_Y,    

def reshape_for_Lstm(data):    
    # reshape for input 
    timesteps=1
    samples=int(np.floor(data.shape[0]/timesteps))

    data=data.reshape((samples,timesteps,data.shape[1]))   #samples, timesteps, sensors     
    return data

    #one hot encode the targets for class prediction/ not for signal prediction
def one_hot(train_Y,val_Y,test_Y):    
    from sklearn.preprocessing import OneHotEncoder
    
    oneHot=OneHotEncoder()
    oneHot.fit(train_Y.reshape(-1,1))
    
    train_Y_Hot=oneHot.transform(train_Y.reshape(-1,1)).toarray()
    val_Y_Hot  =oneHot.transform(val_Y.reshape(-1,1)).toarray()
    test_Y_Hot =oneHot.transform(test_Y.reshape(-1,1)).toarray()
    
    return train_Y_Hot,val_Y_Hot,test_Y_Hot


def model_setup_seq(in_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Dense
    
    model = Sequential()
    model.add(LSTM(32,activation='relu', input_shape=(in_shape[1],in_shape[2]), 
                   return_sequences=True)  )#,
                   # kernel_regularizer=tf.keras.regularizers.L1L2(0.01,0.01)))
    #model.add(Dropout(0.3))
    model.add(LSTM(32,activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
    
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

def plot_training(history,what='loss',saving=False,name='training'):
    fig=plt.figure()
    plt.plot(history[0])
    plt.plot(history[1])
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    if what=='loss':
        plt.title('model loss')
        plt.ylabel('loss')
    elif what=='acc':   
        plt.title('model Acc')
        plt.ylabel('Accuracy')   
    if saving==True:
        fig.savefig( name +'_'+ what + '.png', format='png', dpi=300, transparent=True)


    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    if saving==True:
        fig.savefig( name +'_ACC.png', format='png', dpi=300, transparent=True)  
    plt.show()
    
def plot_signal_hat(Y_test,Y_hat,saving=False,name='results_signal'):
    fig= plt.figure()
    plt.plot(Y_hat)
    plt.plot(Y_test)
    plt.legend(['target','target_predicted'])
    plt.ylabel('Zustand')
    plt.title('Pediction on test data')
    if saving==True:
        fig.savefig( name +'.png', format='png', dpi=300, transparent=True)
    plt.show()
        
def plot_class_hat(Y_hat,Y_test,saving=False,name='results_class'):   
    # HERE WE TRY TO PLOT MULTICOLOR BASED ON PROBABILITY VALUE, BUT IT DOES NOT WOR. IMPUT WELCOME
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    x=np.linspace(1,len(Y_hat),len(Y_hat))

    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1=plt.plot(x,Y_test)
    ax1=plt.scatter(x,Y_hat,c=cm.hot(np.abs(Y_hat)), edgecolor='none') # multicolour print
    plt.legend(['target','target_predicted'])
    if saving==True:
        fig.savefig( name +'.png', format='png', dpi=300, transparent=True)
    plt.show()
       
#%%    
if __name__ == '__main__':
    
#LOAD DATA    
    data,sensorname=read_data('pump_sensor.csv')

#MAP TARGETS TO VALUE    
    encoded_y=Vorverarbeitung_Y(data['machine_status'])
    Values=pd.concat([data[sensorname],encoded_y],axis=1)#.reindex(data.index)

#PREPROCESS DATA   
    Values=manipulate_X(Values, printplot=False); sensorname=Values.keys()[:-1] 

#CREATE WINDOWED DATA
    Future=1

    data_win=series_to_supervised(Values, n_in=Future, n_out=1)
    to_remove_list =['sensor'+str(n)+'(t)' for n in range(1,len(Values.columns)+1)] #now remove all non shifted elements again. so we retreive elements and shifted target
    #to_remove_list_2 =['sensor'+str(n)+'(t-'+ str(i)+')' for n in range(1,len(data_scaled.columns)+1) for i in range(1,Future)] #now remove all non shifted elements again. so we retreive elements and shifted target
    #to_remove_list=to_remove_list_1+to_remove_list_2
    data_y=data_win.iloc[:,-1] #Get the target data out before removing unwanted data
    data_x=data_win.drop(to_remove_list, axis=1) #remove sensors(t)
    data_x.drop(data_x.columns[len(data_x.columns)-1], axis=1, inplace=True)# remove target(t-n)
    
# %%   
#CREATE TRAIN/VAL/TEST SETS
    # We split the data that all sets have at least one error in. But shuffeling is not allowed. Therfore, we do it manually
    
    train_X,train_Y,val_X,val_Y,test_X,test_Y=splitting_and_shape_data(data_x,data_y)
    train_Y_Hot,val_Y_Hot,test_Y_Hot=one_hot(train_Y,val_Y,test_Y)
    
#SCALE THE SETS BETWEEN 0-1
    scaler=MinMaxScaler().fit(train_X)
    train_X=scaler.transform(train_X) 
   
    scaler=MinMaxScaler().fit(val_X)
    val_X=scaler.transform(val_X)  
    
    scaler=MinMaxScaler().fit(test_X)
    test_X=scaler.transform(test_X)  

#RESHAPE THE DATA TO FIT LSTMs samples, timesteps, sensors  FORMAT
    train_X=reshape_for_Lstm(train_X)
    val_X=reshape_for_Lstm(val_X)
    test_X=reshape_for_Lstm(test_X)

# %%
# TRAIN THE MODEL...    
    Train=True
    inputshape_X=(train_X.shape)
    #print(inputshape_X)
    
    if Train==True:
        #model=model_setup_seq(inputshape_X)
        #history = model.fit(train_X, train_Y, epochs=80, batch_size=32, validation_data=(val_X, val_Y), shuffle=False)
    
        model=model_setup_Fapi(inputshape_X)
        history = model.fit(train_X, [train_Y, train_Y_Hot], epochs=20, batch_size=32, validation_data=(val_X, [val_Y,val_Y_Hot]), shuffle=False)
        plot_training([history.history['class_out_loss'],history.history['val_class_out_loss']],
                      what='loss',
                      saving=True,
                      name=('training_'+ str(Future)))  
        plot_training([history.history['class_out_acc'],history.history['val_class_out_acc']],
                      what='acc',
                      saving=True,
                      name=('training_'+ str(Future))) 
        model.save('./model/Pump_LSTM_Fapi_4_'+ str(Future))
        
# ...OR LOAD THE MODELl  
    else:  
        model=tf.keras.models.load_model('./model/Pump_LSTM_Fapi')
        
# %%    
# INFERENCE
    # make a prediction
    [yhat,yclass] = model.predict(test_X)    
    Yclass=[np.argmax(yclass[i],0) for i in range(len(yclass))] # get final class
    
    plot_signal_hat(yhat,test_Y,saving=True, name='Prediction_Signal_fapi3_42_'+ str(Future))
    plot_signal_hat(Yclass,test_Y,saving=True, name='Prediction_class_fapi3_42_'+ str(Future))


# %%
    ##You neede this part only if you want to predict another signal instead of a target. e.g. predict Sensor_42 data.
    #It rescales the signal back to the orignial amplitudes, as it was transformed to 0-1 before. 
    # I leave it in for your convenience.
    
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # # invert scaling for forecast
    
    # inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    # inv_yhat = inv_yhat[:,0]
    # # invert scaling for actual
    # test_Y = test_Y.reshape((len(test_Y), 1))
    # inv_y = np.concatenate((test_Y, test_X[:, 1:]), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:,0]
    # # calculate RMSE
    # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # print('Test RMSE: %.3f' % rmse)

    #plot_results(inv_y,inv_yhat,saving=True)
    # ensure all data is float
#values = values.astype('float32')
 #one-hot-encoding   
 
    
  
    

    