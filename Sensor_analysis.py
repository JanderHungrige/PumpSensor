#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 16:16:12 2021

@author: Jan Werth
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing


def read_data(path):
    dat=pd.read_csv(path)
    #senorname=pd.Series(dat.keys()[2:-1])
    senorname=dat.keys()[2:-1] #geht beides
    return dat, senorname


def explore(data):
    print('Data overview: ')
    print(dat.shape); print()
    print('keys :') ; print(dat.keys()); print()
    print( 'status options: ');  print( dat['machine_status'].unique()); print()
    print (dat['machine_status'].value_counts()); print()
    info=data.describe()
    varianz=pd.DataFrame({'var':data.var()})
    info=pd.concat([info,varianz.transpose()])
    return data.head(), data.tail(), info
    
    #print((dat.isna().sum()))
    
def plotting_stuff(data,plottype,Title, saving=False):
    #plt.plot(dat.loc[:,['sensor_01']])
    fig=plt.figure()
    data.plot(kind=plottype)
    #plt.stem(data)
    plt.title(Title)
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

def plotting_together(data): 
    fig=plt.Figure()
    Values.plot(subplots = True, sharex = True, figsize=(30,55))    
    plt.savefig('Overview.png', format='png', dpi=300, transparent=True)
    plt.show         
    
def manipulate_X(data, printplot=False):
    data['sensor_51'][110000:140000]=data['sensor_50'][110000:140000] # repair sensor 51
    data=data.drop(labels=['sensor_00','sensor_15','sensor_37','sensor_50'],axis=1)#bad sensors
    data=data.drop(labels=['sensor_06','sensor_07','sensor_08','sensor_09'],axis=1)# low varianz#NaNs
    data=data.fillna(method="pad",limit=30)
    data=data.dropna()
    if printplot==True:
        print((data.isna().sum()))
        plotting_stuff((data.isna().sum()[2:-1]),'bar','Manipulated-NaN-drop12filldrop',saving=True)
        

    return data

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

    
    #plotting_stuff((dat.isna().sum())[2:-1],'bar','Raw-NaN',saving=True) # show which sensors have how many NANs
    plotting_stuff(verstehen_std.transpose(),'bar','std',saving=True)# Show std
    plotting_stuff(verstehen_var.transpose(),'bar','var',saving=True)# Show std
    '''
    renoving NaNs
    removing faulty sensors
    removing low varianz sensors
    '''
    #manipulate_X(dat, printplot=True)
    
    '''
    Plotting the sensor signals raw
    '''
    # for i in senorname:
    #     plotting_stuff(dat[i],'line',str(i))

    '''
    Plotting sensor and label together
    '''
    encoded_y=Vorverarbeitung_Y(dat['machine_status'])
    Values=pd.concat([dat[senorname],encoded_y],axis=1)#.reindex(dat.index)
    plotting_merged(dat[senorname],encoded_y, senorname,saving=True)# plot each singal with target
    #plotting_together(Values) #plot all signals together with target

    
'''
Some sensor information
   shit sensors:
       
       packs:
           1,9
           24678 14 19 20 21 22 23 24 25 26 27 28 29 - 36(Low varianze mean 1)
           38 39  41 42 43 44 -49 (low amplitude+ varianze mean 0)
           3 14 16 17 18  more noise thinner (mean 1) 
           13 more noise thinner (mean 0)
           5 10 11 12 higher varianz/thicker
           
           
       0,37,15,50 (miising data), really bad 
       use 50 to repair 51
       
       37 :clipped as it seems
       19,24: aslo clipped but not to bad
18,17,16 even less bad

44,43,42,41,40,39,38: Now much change as it seems. Low ampliude 0-1

'''
    