#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:16:12 2021

@author: Jan Werth
This script uses the data from https://ga-data-cases.s3.eu-central-1.amazonaws.com/pump_sensor.zip

Here we take a look at the data to see what is usefull information and how to manipulate the data.

ToDo: 
   next step would be to remove outliers with z-score
   
   also try not to remove sensors 'sensor_06','sensor_07','sensor_08','sensor_09'. How would the results change
      
   optimize line 96  data=data.drop(labels=['sensor_06','sensor_07','sensor_08','sensor_09'],axis=1)
   Here we create a copy which might lead to crashes
   
   Look furhter into the future than 10min
   
   Try GRU / Transformers for better performance (and maybe Random forrest for embedded devices)

"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing


def read_data(path):
    dat=pd.read_csv(path)
    #senorname=pd.Series(dat.keys()[2:-1])
    senorname=dat.keys()[2:-1] #geht beides
    return dat, senorname

# def counting_target(data):
#     laenge=data['machine_status'].Count.groupby(
#         (data['machine_status'] != data['machine_status'].Count.shift()).cumsum()).transform('size')*data['machine_status'].Count

#     return laenge

def explore(data):
    print('Data overview: ')
    print(data.shape); print()
    print('keys :') ; print(data.keys()); print()
    print( 'status options: ');  print( data['machine_status'].unique()); print()
    print (data['machine_status'].value_counts()); print()
    #print((data.isna().sum())[2:-1]); print()
    info=data.describe()
    varianz=pd.DataFrame({'var':data.var()})
    info=pd.concat([info,varianz.transpose()])
    return data.head(), data.tail(), info
        
def manipulate_X(data, printplot=False):
    data=data.drop(labels=['sensor_15'],axis=1)#bad sensors
    data=data.drop(labels=['sensor_00'],axis=1)#bad sensors

    data['sensor_51'][110000:140000]=data['sensor_50'][110000:140000] # repair sensor 51
    data=data.drop(labels=['sensor_50'],axis=1)#bad sensors

   # data=data.drop(labels=['sensor_00','sensor_15','sensor_37','sensor_50'],axis=1)#bad sensors
    data=data.drop(labels=['sensor_06','sensor_07','sensor_08','sensor_09'],axis=1)# low varianz#NaNs
    data=data.fillna(method="pad",limit=30)
    data=data.dropna()
    if printplot==True:
        print((data.isna().sum()))
        plotting_stuff((data.isna().sum()[2:-1]),'bar','fill_nan',saving=True)
        
    return data    
    
def plotting_stuff(data,plottype,Title, saving=False):
    #plt.plot(dat.loc[:,['sensor_01']])
    fig=plt.figure()
    data.plot(kind=plottype)
    #plt.stem(data)
    plt.title(Title)
    #plt.xticks(rotation=45)
    if saving==True:
        plt.savefig(Title+'.png', format='png', dpi=300, transparent=True)    
    #fig.show()
    
def plotting_merged(data, encoded_y, senorname, saving=False):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_dat = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    for i in senorname:
        fig=plt.figure()
        ax=scaled_dat[i].plot.line()
        encoded_y.plot(ax=ax)
        plt.title('together_' + str(i) )
        plt.legend(['sensor','target'])
        if saving==True:
            fig.savefig('Sensor_'+str(i)+'.png', format='png', dpi=300, transparent=True)
        plt.show()    

def plotting_together(Values): 
    fig=plt.Figure()
    Values.plot(subplots = True, sharex = True, figsize=(30,55))    
    plt.savefig('Overview.png', format='png', dpi=300, transparent=True)
    plt.show      
    
def plot_Y(data, col='target', saving=False, name='target'):
    import numpy as np
    y=data[col]; x=np.linspace(1,len(y),len(y))
    plt.plot(x,y)
    plt.ylabel('class')
    plt.title('Target')
    labels = ['Normal','Broken','Recovering']
    if col=='target':
        plt.yticks([1,0,2], labels, rotation='vertical')
    elif col=='machine_status':
        plt.yticks([0,1,2], labels, rotation='vertical')
    if saving==True:
        plt.savefig(name+'.png', format='png', dpi=300, transparent=True)
    plt.show()   


def Vorverarbeitung_Y(data):
    from sklearn import preprocessing

    #Label Mapping
    le = preprocessing.LabelEncoder()
    le.fit(data)
    encoded_y=le.transform(data)
    #Get the Label map
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)
    return pd.DataFrame(encoded_y,columns=['target'])


if __name__ == '__main__': 
    
    dat,senorname=read_data('pump_sensor.csv')
    
    Kopf,Schwanz, verstehen=explore(dat)
    
    '''
    Checking for Nans, 
    Checking for std
    '''
    verstehen_std=verstehen.loc[['std']][senorname]
    verstehen_var=verstehen.loc[['var']][senorname]

    plot_Y(dat,col='machine_status',saving=False,name='Klassen')
    plotting_stuff((dat.isna().sum())[2:-1],'bar','Raw-NaN',saving=False) # show which sensors have how many NANs
    plotting_stuff(verstehen_std.transpose(),'bar','std',saving=True)# Show std
    plotting_stuff(verstehen_var.transpose(),'bar','var',saving=True)# Show std
    
    '''
    renoving NaNs
    removing faulty sensors
    removing low varianz sensors
    '''
    manipulate_X(dat, printplot=True)
    
    '''
    Plotting the sensor signals raw
    '''
    for i in senorname:
         plotting_stuff(dat[i],'line',str(i))

    '''
    Plotting sensor and label together
    '''
    encoded_y=Vorverarbeitung_Y(dat['machine_status']);   
    #laenge=counting_target(dat)

    plot_Y(encoded_y,col='target', saving=True , name='Klassen')

    Values=pd.concat([dat[senorname],encoded_y],axis=1)#.reindex(dat.index)
    plotting_merged(dat[senorname],encoded_y, senorname,saving=True)# plot each singal with target
    plotting_together(Values) #plot all signals together with target

        