#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:03:11 2021

@author: base
"""
import matplotlib.pyplot as plt

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